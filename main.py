import matplotlib.pyplot as plt
import base64
from fastapi import Request, FastAPI, HTTPException
from pydantic import BaseModel
import logging
import pandas as pd
import joblib
import re
import sklearn
from pymystem3 import Mystem
import random
from redis import Redis
from fastapi.responses import JSONResponse
from io import BytesIO


logging.basicConfig(level=logging.ERROR)

app = FastAPI()

model = joblib.load("base_logreg_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
mystem = Mystem()

df = pd.read_csv('data_50k_preproc.csv')


class News(BaseModel):
    text: str


def lemmatize_text_russian(text):
    text = str(text)
    text = re.sub(r'^[\r\n\s]+|[\r\n\s]+$', '', text)
    text = re.sub(r'\[|\]', '', text)
    lemmatized_tokens = mystem.lemmatize(text.lower())
    return " ".join([token for token in lemmatized_tokens if token.strip()])


@app.get("/predict/")
async def get_predict():
    return {"message": "GET request to /predict/ is not supported. Please use POST method to send the news text."}


@app.post("/predict/")
async def predict(news: News):
    processed_text = lemmatize_text_russian(news.text)
    predicted_tag = model.predict([processed_text])[0]
    predicted_word = label_encoder.inverse_transform([predicted_tag])[0]
    return {"Предсказанный тэг": predicted_word, 'обработанный текст: ': processed_text}


@app.get("/random_row/")
async def get_random_row():
    random_index = random.randint(0, len(df) - 1)
    random_row = df.iloc[random_index].to_dict()
    return random_row


redis = Redis(host="localhost", port=6379)


@app.get("/tag-time-plot/{tag}")
async def tag_time_plot(request: Request, tag: str):

    cache_key = f"tag-time-series-{tag}"
    cached_image = redis.get(cache_key)

    if cached_image is None:
        df = pd.read_csv("data_50k_preproc.csv")
        df = df[df["most_popular_tag"] == tag]
        df['date and time'] = pd.to_datetime(df['date and time'])
        df['month'] = df['date and time'].dt.to_period('M').astype(str)

        tag_counts = []
        months = []

        for month, group in df.groupby('month'):
            tag_count = group[['most_popular_tag']].eq(tag).sum().sum()
            tag_counts.append(tag_count)
            months.append(month)

        result_df = pd.DataFrame({'months': months, 'tag_counts': tag_counts})

        plt.figure(figsize=(10, 6))
        plt.plot(result_df['months'], result_df['tag_counts'], linestyle='-', color='purple', label='Tag Occurrences')
        plt.title(f'Total occurrences of tag "{tag}" per month')
        plt.xlabel('Month')
        plt.ylabel('Total occurrences')
        plt.gca().get_xaxis().set_major_locator(plt.MaxNLocator(nbins=10))

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        redis.set(cache_key, image_base64, ex=3600)
        return JSONResponse(content={"image_base64": image_base64})

    else:
        return JSONResponse(content={"image_base64": cached_image.decode('utf-8')})


class URLRequest(BaseModel):
    url: str


class TextResponse(BaseModel):
    text: str


@app.post("/get_text/")
async def get_text(url_request: URLRequest):
    if url_request.url in df['url'].values:
        text = df.loc[df['url'] == url_request.url, 'text'].values[0]
        return TextResponse(text=text)
    else:
        return {"error": "URL not found"}


@app.get("/popular-tags/")
async def get_popular_tags(date: str):

    date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_match = re.search(date_pattern, date)
    try:
        input_date = pd.to_datetime(str(date_match.group(0)))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please provide date in 'YYYY-MM-DD' format.")

    df['date and time'] = pd.to_datetime(df['date and time'])
    filtered_df = df[df['date and time'].dt.date == input_date.date()]

    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="Data not found for the specified date.")

    tags_series = filtered_df['most_popular_tag']
    tag_counts = tags_series.value_counts()

    popular_tags = tag_counts.head(10).index.tolist()

    return {"popular_tags": popular_tags}


@app.get("/latest_news/{tag}")
async def get_latest_news(tag: str):
    latest_news = df[df['most_popular_tag'] == tag].tail(2)
    if latest_news.empty:
        raise HTTPException(status_code=404, detail="Новости с данным тэгом не найдены")
    latest_news_list = latest_news.to_dict(orient='records')
    return latest_news_list


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
