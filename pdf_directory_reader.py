import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from keybert import KeyBERT
from llama_index import SimpleDirectoryReader, Document
from llama_index.readers.base import BaseReader
from llmsherpa.readers import LayoutPDFReader
from pydantic import PrivateAttr


class PDFDirectoryReader(SimpleDirectoryReader):
    """PDF Directory Reader extends SimpleDirectoryReader

    It's based on LayoutPDFReader for the PDF Parsing and KeyBERT for keywords/keyphrases extraction.
    """

    def __init__(self,
                 llm_sherpa_url: str,
                 **kwargs):
        """
        :param llm_sherpa_url: API url for LLM Sherpa. Use custom url for your private instance here
        :param keyphrase_ngram_range: Optional, float between 0 (min) and 1 (max). Default 0.9
        :param keyphrase_diversity: Optional, keyphrases min/max length. Default (1,1). E.g. (1,3) = min 1, max 3
        :param with_header_cleansing: Optional, set to False to disable the cleansing of the PDF headers
        :param file_extractor: Optional, dict with file extractors
        """
        file_extractor = kwargs.pop('file_extractor') if 'file_extractor' in kwargs else {}
        with_header_cleansing = kwargs.pop('with_header_cleansing') if 'with_header_cleansing' in kwargs else True
        file_extractor['.pdf'] = SmarterPDFReader(llm_sherpa_url, with_header_cleansing, **kwargs)
        kwargs['file_extractor'] = file_extractor
        super().__init__(**kwargs)


class SmarterPDFReader(BaseReader):
    """
    Internal class used to parse a PDF
    """

    layout_pdf_reader: LayoutPDFReader = PrivateAttr()
    keyword_model: KeyBERT = PrivateAttr()
    keyphrase_ngram_range = PrivateAttr()  #
    keyphrase_diversity = PrivateAttr()

    def __init__(self,
                 llm_sherpa_url: str,
                 with_header_cleansing: bool,
                 keyphrase_ngram_range: Optional[Tuple] = (1, 1),
                 keyphrase_diversity: Optional[float] = 0.9,
                 **kwargs):
        super().__init__()
        self.layout_pdf_reader = LayoutPDFReader(llm_sherpa_url)
        self.keyword_model = KeyBERT()
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.keyphrase_diversity = keyphrase_diversity
        self.header_cleaner = HeaderCleansing() if with_header_cleansing else None
        self.with_header_cleansing = with_header_cleansing

    def load_data(
            self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        abs_path = file.absolute().as_posix()
        doc = self.layout_pdf_reader.read_pdf(abs_path)
        docs = []

        last_header = None

        for idx, chunk in enumerate(doc.chunks()):
            parent_text = chunk.parent_text()
            text = chunk.to_context_text(include_section_info=False)

            keywords = self.keyword_model.extract_keywords(text,
                                                           keyphrase_ngram_range=self.keyphrase_ngram_range,
                                                           use_mmr=True,
                                                           diversity=self.keyphrase_diversity)

            header = self.header_cleaner.cleanse(parent_text,
                                                 last_header) if self.with_header_cleansing else parent_text
            last_header = header

            metadata = {
                'header': header,
                'chunk_index': idx,
                'file_name': file.name,
                'keywords': keywords
            }

            docs.append(PDFDocument(text=text, metadata=metadata))

        return docs


class PDFDocument(Document):
    """
    Document representing a PDF
    """

    def __init__(self,
                 text: str,
                 metadata: Dict):
        super().__init__(text=text, metadata=metadata)

    def __str__(self) -> str:
        index = self.chunk_index()
        header = self.metadata['header']
        keywords = self._keywords_as_str()
        sep = f'{"-" * 32}'
        large_sep = f'{"=" * 32}'
        return f'{large_sep}\nchunk {index}\n{header}\n{keywords}\n{sep}\n{self.text}\n'

    def chunk_index(self) -> int:
        return self.metadata['chunk_index']

    def _keywords_as_str(self, separator: Optional[str] = ' / '):
        return separator.join([f'({k[0]}, {k[1]})' for k in self.metadata['keywords']])


class HeaderCleansing:
    """
    Internal class used to cleanse the headers in a PDF
    """

    def cleanse(self, header: str, current: str) -> str:
        multi_level_header = re.compile('.+ > .+')
        if current is None or multi_level_header.search(header) is None:
            return header.strip()
        tokens_current = self._split_token(current)
        tokens_header = self._split_token(header)
        tokens_header = self._enrich_header(tokens_header, tokens_current)
        merge = self._join(tokens_header, tokens_current)
        return merge if len(merge) > 0 else header

    @staticmethod
    def _split_token(header: str) -> list[str]:
        starts_with_number = re.compile('^((\d+\.?)+)(.*)')
        strip_page_numbers = re.compile('(.*)( +.? \d+)')
        tokens = []
        last_index = ''
        for token in header.split(" > "):
            result = starts_with_number.search(token)
            if result is None:
                continue
            index = result.group(1)
            dirty_title = result.group(3).strip()
            result_inner = strip_page_numbers.search(dirty_title)
            title = dirty_title if result_inner is None else result_inner.group(1)
            if len(last_index) < len(index):
                tokens.append(f'{index} {title}')
                last_index = index
            else:
                break
        return tokens

    @staticmethod
    def _enrich_header(tokens_header: list[str], tokens_current: list[str]) -> list[str]:
        for i, _ in enumerate(tokens_current):
            if len(tokens_header) <= i:
                tokens_header = tokens_current
                break
            elif tokens_header[i] == tokens_current[i] or len(tokens_header[i].split()[0].split(".")) == len(
                    tokens_current[i].split()[0].split(".")):
                break
            tokens_header.insert(i, tokens_current[i])
        return tokens_header

    @staticmethod
    def _join(tokens_header: list[str], tokens_current: list[str]) -> str:
        if len(tokens_header) == len(tokens_current):
            return " > ".join(tokens_header)
        else:
            return " > ".join(tokens_header if len(tokens_current) > len(tokens_header) else tokens_header)
