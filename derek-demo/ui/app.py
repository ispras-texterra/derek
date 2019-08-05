import os
import sys
from collections import Counter

from flask import Flask, render_template, request
from pymongo import MongoClient, DESCENDING, ASCENDING
from bson import ObjectId
from bson.errors import InvalidId

app = Flask(__name__)
db = None


def get_news_on_page(page_num, page_size=10):
    return list(db.find().sort('time', DESCENDING).skip(page_num * page_size).limit(page_size)), \
        page_num > 0, (page_num + 1) * page_size < db.count()


def get_news_by_id(news_id):
    try:
        news = db.find_one({'_id': ObjectId(news_id)})
        prev_news = list(db.find({'time': {'$gt': news['time']}}).sort('time', ASCENDING).limit(1))
        next_news = list(db.find({'time': {'$lt': news['time']}}).sort('time', DESCENDING).limit(1))
        return include_spans_and_types_into_news(news),\
            str(prev_news[0]['_id']) if prev_news else None,\
            str(next_news[0]['_id']) if next_news else None
    except InvalidId:
        return None, None, None


known_entity_types = None


def include_spans_and_types_into_news(news):
    def between_entities(spans_per_par, text):
        for idx, part in enumerate(text.split('\n\n')):
            if idx > 0:  # start new paragraph after newlines
                spans_per_par.append([])
            spans_per_par[-1].append({'text': part, 'color': ''})

    def color_for_type(ent_type):
        PALETTE = ['#f7b32b', '#fcf6b1', '#e5e059', '#bdd358', '#ceeddb',
                   '#ffb86f', '#c5d86d', '#f9cff2', '#ddfbd2', '#33a1fd']
        try:
            idx = known_entity_types.index(ent_type) % (len(PALETTE) - 1)
        except ValueError:
            idx = len(PALETTE) - 1  # last is reserved for unknown types
        return PALETTE[idx]

    text, entities = news['text'], news['entities']
    prev_start = 0
    spans = [[]]
    ent_types_cnt = Counter()
    for ent in entities:
        start, end, ent_type = ent['start'], ent['end'], ent['type']
        if start > prev_start:
            between_entities(spans, text[prev_start:start])
        spans[-1].append({'text': text[start:end], 'color': color_for_type(ent_type)})
        prev_start = end
        ent_types_cnt[ent_type] += 1

    if len(text) > prev_start:
        between_entities(spans, text[prev_start:])

    ent_types = []
    for ent_type, cnt in ent_types_cnt.most_common():
        ent_types.append({'type': ent_type, 'color': color_for_type(ent_type), 'count': cnt})
    return {**news, 'spans': spans, 'ent_types': ent_types}


@app.route("/")
def index():
    page_num = request.args.get('page', '1')
    try:
        page_num = int(page_num)
    except ValueError:
        return f"Invalid page number: {page_num}", 400
    if page_num < 1:
        return f"Invalid page number: {page_num}", 400

    news_list, has_prev_page, has_next_page = get_news_on_page(page_num - 1)
    if not news_list:
        return "News not found", 404

    return render_template("index.html", news_list=news_list,
                           prev_page=(page_num - 1) if has_prev_page else None,
                           next_page=(page_num + 1) if has_next_page else None)


@app.route("/news/<news_id>")
def news_info(news_id: str):
    news, prev_id, next_id = get_news_by_id(news_id)
    if news is None:
        return "News not found", 404
    return render_template("news.html", news=news, prev_id=prev_id, next_id=next_id)


if __name__ == "__main__":
    mongo_uri = os.environ.get("MONGODB_URI", None)
    mongo_db_name = os.environ["MONGODB_DB_NAME"]
    db = MongoClient(mongo_uri)[mongo_db_name]['processed']
    known_entity_types = sys.argv[1:]

    app.run(host='0.0.0.0')
