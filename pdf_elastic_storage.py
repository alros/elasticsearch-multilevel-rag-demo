from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from llm_summary import LLMSummary
from pdf_directory_reader import PDFDirectoryReader

llmsherpa_api_url = 'http://localhost:5010/api/parseDocument?renderFormat=all'

elasticsearch_url = 'http://localhost:9200'
sentence_transformers_model_name = 'all-MiniLM-L6-v2'


class PDFElasticStorage:
    """
    Represents a storage based on Elasticsearch with the capability to store documents in two levels.
    - Level 1: contains summaries of several chunks.
    - Level 2: contains the chunks.
    The research works on level 1 first, and uses the result to filter the research on level 2.
    """

    def __init__(self,
                 index_name_summary: str = 'summary',
                 index_name_chunks: str = 'chunks',
                 max_chunks_per_summary: int = 700):
        """
        :param index_name_summary: Optional, name of the index to store summaries. Default 'summary'
        :param index_name_chunks: Optional, name of the index to store chunks. Default 'chunks'
        :param max_chunks_per_summary: Optional approximate length of a block of chunks before being summarised, in
                                       words. Default 700.
        """
        self._index_name_summary = index_name_summary
        self._index_name_chunks = index_name_chunks
        self._max_chunks_per_summary = max_chunks_per_summary
        self._llm_summary = LLMSummary()
        self._elasticsearch = ElasticsearchWrapper()

    def ingest(self, folder: str):
        """
        Ingest all the documents in a folder.
        :param folder: path
        :return: None
        """
        reader = PDFDirectoryReader(llm_sherpa_url=llmsherpa_api_url,
                                    input_dir=folder,
                                    recursive=True,
                                    with_header_cleansing=True)
        for document_chunk_list in reader.iter_data():
            for chunks in self._chunks(document_chunk_list):
                text = '\n'.join([d.text for d in chunks])
                summary = self._llm_summary.summarise(text)
                doc_id = self._elasticsearch.add_document(document={
                    "text": summary
                }, index=self._index_name_summary)
                for chunk in chunks:
                    self._elasticsearch.add_document(document={
                        "text": chunk.text,
                        "parent_id": doc_id,
                    }, index=self._index_name_chunks)

    def find(self,
             query: str,
             top_summary: int = 3,
             top_chunks: int = 10,
             debug: bool = False):
        """
        finds chunks
        :param query: any text
        :param top_summary: Optional. Find up to n summaries. Default 3.
        :param top_chunks: Optional. Find up to n chunks. Default 10.
        :param debug: Optional. Print debug information. Default False.
        :return: list of chunks
        """
        summary_results = self._elasticsearch.find(query=query,
                                                   index=self._index_name_summary,
                                                   top=top_summary)
        ids = [r['_id'] for r in summary_results['hits']['hits']]
        chunk_results = self._elasticsearch.find(query=query,
                                                 index=self._index_name_chunks,
                                                 top=top_chunks,
                                                 parent_ids=ids)
        if debug:
            print(f'parents:')
            for parent in summary_results['hits']['hits']:
                print(f'- id:{parent["_id"]}')
                print(f'  score:{parent["_score"]}')
            print(f'\n  chunks:\n')
            for chunk in chunk_results['hits']['hits']:
                print(f'  - id={chunk["_id"]}')
                print(f'    parent_id={chunk["_source"]["parent_id"]}')
                print(f'    score={chunk["_score"]}')
                print(f'    text=\n"""\n{chunk["_source"]["text"]}\n"""\n')

        return [chunk['_source']['text'] for chunk in chunk_results['hits']['hits']]

    def reset(self):
        """
        reset the db.
        :return: None
        """
        self._elasticsearch.delete_index(self._index_name_summary)
        self._elasticsearch.delete_index(self._index_name_chunks)

    def _chunks(self, seq):
        i = 0
        j = 0
        chunks = []
        while i < len(seq):
            text = seq[i].text
            i = i + 1
            while len(text.split(' ')) < self._max_chunks_per_summary and i < len(seq):
                text = text + seq[i].text + ' '
                i = i + 1
            chunks.append(seq[j:i])
            j = i
        return chunks


class ElasticsearchWrapper:
    """
    Internal class used to model Elasticsearch
    """

    def __init__(self):
        self._es_instance = None
        self._model = None

    def _es(self):
        if self._es_instance is None:
            self._es_instance = Elasticsearch(elasticsearch_url)
        return self._es_instance

    def create_index(self, index: str):
        self._es().indices.create(index=index, mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                }
            }
        })

    def delete_index(self, index: str):
        self._es().indices.delete(index=index, ignore_unavailable=True)

    def add_document(self, document: dict, index: str) -> str:
        document = {
            **document,
            'embedding': self._get_embedding(document['text']),
        }
        response = self._es().index(index=index, body=document)
        return response['_id']

    def find(self,
             query: str,
             index: str,
             top: int = 10,
             parent_ids: list[str] = None):
        query_filter = {'filter': []}
        if parent_ids is not None:
            query_filter['filter'].append({
                "terms": {
                    "parent_id.keyword": parent_ids
                }
            })
        return self._es().search(index=index,
                                 knn={
                                     'field': 'embedding',
                                     'query_vector': self._get_embedding(query),
                                     'num_candidates': top,  # shard level
                                     'k': top,  # total
                                     **query_filter
                                 })

    def _get_embedding(self, text: str):
        if self._model is None:
            self._model = SentenceTransformer(sentence_transformers_model_name)
        return self._model.encode(text)
