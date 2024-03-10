from typing import List

from llama_index import QueryBundle
from llama_index.core.base_retriever import BaseRetriever
from llama_index.schema import NodeWithScore, Document


class NullRetriever(BaseRetriever):
    """
    A retriever that does not return anything. Essentially a placeholder for future extensions
    """

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [NodeWithScore(node=Document(text='', node_id='', metadata={'file_name': ''}), score=0)]
