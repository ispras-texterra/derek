import argparse
import requests
import os

from pymongo import MongoClient, DESCENDING

BATCH_SIZE = 100


def process_new_in_collection(raw_items_coll, processed_items_coll, server_url, need_re):
    num_processed = 0
    last_id = None
    stop = False

    while not stop:
        query = {'_id': {'$lt': last_id}} if last_id is not None else {}
        raw_batch = list(raw_items_coll.find(query).sort('_id', DESCENDING).limit(BATCH_SIZE))

        if not raw_batch:
            # no more items
            return num_processed

        last_id = raw_batch[-1]['_id']
        raw_text_batch = [{"text": elem["text"]} for elem in raw_batch]
        response_batch = requests.post(
            server_url, json=raw_text_batch, params={"entities": "1", "relations": "1" if need_re else "0"}).json()

        processed_batch = []

        for raw_item, response in zip(raw_batch, response_batch):
            if processed_items_coll.find_one({'_id': raw_item['_id']}):
                # item already processed
                stop = True
                break

            processed_item = dict(raw_item)
            processed_item['source'] = raw_items_coll.name
            processed_item.update(response)

            processed_batch.append(processed_item)
            num_processed += 1

        if processed_batch:
            processed_items_coll.insert_many(processed_batch)

    return num_processed


def main():
    argparser = argparse.ArgumentParser(description='Processor of collected news in DEREK-demo')
    argparser.add_argument('server_url', type=str, help='DEREK server address')
    argparser.add_argument('output_collection', type=str, help="Collection to store processed news")
    argparser.add_argument('collections', type=str, nargs="+", help="Collections with collected news to process")
    argparser.add_argument('-need_rels', action='store_true', help="Ask server for rel_ext")

    args = argparser.parse_args()

    with MongoClient(os.environ.get("MONGODB_URI", None)) as client:
        db = client[os.environ["MONGODB_DB_NAME"]]
        for name in args.collections:
            num_processed = process_new_in_collection(
                db[name], db[args.output_collection], args.server_url, args.need_rels)
            print(f'Processed {num_processed} new items in collection {name}')


if __name__ == '__main__':
    main()
