from pymongo import MongoClient, UpdateOne
import os

import pymongo
from dawn_vok.db.mongo_ind import MongoIndexes
class MongoUtils:
    MONGO_URI = os.getenv('MONGO_URI') or 'mongodb://127.0.0.1:27017'
    print(MONGO_URI)
    client = MongoClient(MONGO_URI)
    allowed_dbs = ['meta_data', 'sensor_data', 'raw_data', 'models', 'embeddings']
    
    

    @staticmethod
    def get_collection(db_name, collection_name, index_list=None):
        if db_name not in MongoUtils.allowed_dbs:
            raise ValueError(f"Database {db_name} is not allowed")
        db = MongoUtils.client[db_name]
        allowed_collections =  MongoIndexes.total_collections.get(db_name, {})
        if collection_name not in allowed_collections:
            raise ValueError(f"Collection {collection_name} is not allowed in {db_name}")
        else:
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
            index_list = allowed_collections.get(collection_name, index_list)
        collection = db[collection_name]
        if index_list is not None:
            MongoUtils.create_indexes(collection, index_list)
        return collection

    @staticmethod
    def get_db(db_name):
        return MongoUtils.client[db_name]


    @staticmethod
    def create_indexes(collection, index_list):
        existing_indexes = collection.index_information()

        for index_def in index_list:
            # Unpack (fields, options) or default to no options
            if isinstance(index_def, tuple) and len(index_def) == 2:
                fields, options = index_def
            else:
                fields = index_def
                options = {}

            # Build expected index name (e.g., field_1_field2_-1)
            index_name = '_'.join(f"{field[0]}_{field[1]}" for field in fields)

            if index_name not in existing_indexes:

                collection.create_index(fields, **options)
    
    @staticmethod
    def update_many(db_name, collection_name, data, index_field='_id'):
        col = MongoUtils.get_collection(db_name, collection_name)
        bulk_updates = []
        for doc in data:
            if not isinstance(doc, dict):
                doc = doc.to_dict()
            if index_field not in doc:
                raise ValueError(f"Document {doc} does not have an index field")
            doc_id = doc[index_field]
            # Create an update operation for each document.
            bulk_updates.append(
                UpdateOne({'_id': doc_id}, {'$set': doc}, upsert=True)
            )
        if bulk_updates:
            col.bulk_write(bulk_updates)
            print(f"Updated {len(bulk_updates)} documents in {db_name}.{collection_name}")
        else:
            print(f"No updates to perform in {db_name}.{collection_name}")
        return True
    
    @staticmethod

    def atm_increment_value(
        db_name,
        collection_name,
        document_id,
        variable_path,
        inc,
        upsert_di=None
    ):
        """
        Atomically increments a specified numerical field within a MongoDB document
        and returns the value of the field *before* the increment.

        Additionally, if 'upsert_di' is provided and the document doesn't exist,
        the document will be created (upserted) with the additional data contained
        in 'upsert_di', along with the incremented field.

        Args:
            db_name (str): The name of the target database.
            collection_name (str): The name of the target collection.
            document_id: The _id of the document to update.
            variable_path (str): The name of the field to increment (supports dot notation for nested fields).
            inc (int or float): The amount to increment the field by.
            upsert_di (dict, optional): Additional data to set if a new document is created during upsert.
                                        These fields will be inserted only if the document does not exist.

        Returns:
            int or float: The numerical value of the specified field *before* it was incremented.
                        Returns 0 if the field did not exist prior to the increment or if the document was upserted.

        Raises:
            TypeError: If 'inc' is not an int or float or if an existing field is non-numeric.
            ValueError: If 'variable_path' or 'document_id' is invalid.
            RuntimeError: If the DB operation fails (and no upsert is intended) or other unexpected errors occur.
        """
        # Validate parameters
        if not isinstance(inc, (int, float)):
            raise TypeError("inc must be an integer or float.")
        if not variable_path or not isinstance(variable_path, str):
            raise ValueError("variable_path must be a non-empty string.")
        if document_id is None:
            raise ValueError("document_id cannot be None.")

        try:
            # Get the MongoDB collection object
            collection = MongoUtils.get_collection(db_name=db_name, collection_name=collection_name)

            # Build the update document.
            # The $inc operator is used to increment the target field.
            update_doc = {"$inc": {variable_path: inc}}
            if upsert_di is not None:
                # Use $setOnInsert to specify additional data that should be set upon upsert (document creation).
                update_doc["$setOnInsert"] = upsert_di

            # Set upsert flag: enable upsert if additional insert data is provided.
            upsert_flag = True if upsert_di is not None else False

            # Atomically find the document, apply the update, and return the state of the document before the update.
            result_doc_before = collection.find_one_and_update(
                {"_id": document_id},
                update_doc,
                projection={variable_path: 1},
                return_document=pymongo.ReturnDocument.BEFORE,
                upsert=upsert_flag
            )

            # If result_doc_before is None:
            # - Without upsert, that means no document was found.
            # - With upsert enabled, a new document was inserted and there is no "before" state.
            if result_doc_before is None:
                if not upsert_flag:
                    raise RuntimeError(f"Document with id '{document_id}' not found in DB '{db_name}.{collection_name}'.")
                # When upserting, if no document existed, the effective previous value is assumed to be 0.
                return 0

            # Extract the current value from the returned document,
            # following the dot notation provided in variable_path.
            value_before = result_doc_before  # Starting with the complete document (or projection)
            path_keys = variable_path.split('.')

            found = True
            for key in path_keys:
                if isinstance(value_before, dict) and key in value_before:
                    value_before = value_before[key]
                else:
                    found = False
                    break

            if not found:
                # Field did not exist before $inc was applied.
                value_before_numeric = 0
            elif isinstance(value_before, (int, float)):
                # Field existed and its value is numeric.
                value_before_numeric = value_before
            else:
                # Field exists but is not numeric.
                raise TypeError(f"Field '{variable_path}' in document '{document_id}' existed but was not numeric before increment attempt.")

            return value_before_numeric

        except Exception as e:
            # Re-raise expected exceptions and wrap any unexpected errors.
            if isinstance(e, (RuntimeError, ValueError, TypeError)):
                raise e
            else:
                raise RuntimeError(f"Failed to atomically increment '{variable_path}' for document '{document_id}'. Error: {e}") from e

if __name__ == '__main__':
    index_list =     [
        ([('sensor_id', 1)], {'unique': True}),
        ([('timestamp', -1)], {}),  # non-unique
        [('sensor_id', 1), ('timestamp', -1)]  # default (non-unique) form still supported
    ]
    collection = MongoUtils.get_collection('meta_data', 'sensor_md')
    MongoUtils.create_indexes(collection, index_list)
