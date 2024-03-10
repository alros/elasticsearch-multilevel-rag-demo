from llama_index import ServiceContext, ChatPromptTemplate
from llama_index.callbacks import LlamaDebugHandler, CallbackManager, CBEventType
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index.llms import Ollama
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from null_retriever import NullRetriever

llm_model = 'mistral'

system_prompt = """\
You are scientific service able to summarise large texts"""

user_prompt = """\
Context information is below.
<context>
{query_str}
</context>
Summarise the context."""


class LLMSummary:
    """
    Class used to summarise text with Ollama.
    """

    def __init__(self):
        self._engine = None

    def summarise(self, text: str, debug: bool = False) -> str:
        """
        Summarise text
        :param text: content to summarise
        :param debug: Optional. Default False
        :return: summarised text
        """
        engine = self._get_instance()
        response = engine.query(text)
        if debug:
            LLMSummary._print_debug(llama_debug=self._llama_debug, response=response)
        return response.response

    def _get_instance(self) -> RetrieverQueryEngine:
        if self._engine is None:
            llm = Ollama(model=llm_model)

            self._llama_debug = LlamaDebugHandler(print_trace_on_end=True)
            callback_manager = CallbackManager([self._llama_debug])
            service_context = ServiceContext.from_defaults(llm=llm,
                                                           embed_model="local",
                                                           callback_manager=callback_manager)
            vector_retriever_chunk = NullRetriever()
            text_qa_template = ChatPromptTemplate([
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=system_prompt,
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=user_prompt,
                ),
            ])
            self._engine = RetrieverQueryEngine.from_args(
                vector_retriever_chunk,
                service_context=service_context,
                verbose=True,
                response_mode=ResponseMode.COMPACT,
                text_qa_template=text_qa_template
            )
        return self._engine

    @staticmethod
    def _print_debug(llama_debug: LlamaDebugHandler, response):
        event_pairs = llama_debug.get_event_pairs(CBEventType.LLM)
        print("\n" + ("=" * 20) + " RESPONSE " + ("=" * 20) + "\n")
        for node in response.source_nodes:
            print(f'{node.node_id}: score {node.score} - {node.node.metadata["file_name"]}\n\n')
        print("\n" + ("=" * 20) + " /RESPONSE " + ("=" * 20) + "\n")
        print("\n" + ("=" * 20) + " DEBUG " + ("=" * 20) + "\n")
        for event_pair in event_pairs:
            print(event_pair[0])
            print(event_pair[1].payload.keys())
            print(event_pair[1].payload["response"])
        print("\n" + ("=" * 20) + " /DEBUG " + ("=" * 20) + "\n")
