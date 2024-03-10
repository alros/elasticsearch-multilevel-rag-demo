from pdf_elastic_storage import PDFElasticStorage

pdf_multilevel_loader = PDFElasticStorage()
pdf_multilevel_loader.reset()
pdf_multilevel_loader.ingest('pdf')

results = pdf_multilevel_loader.find('this is my query',
                                     top_summary=10,
                                     top_chunks=3,
                                     debug=True)

print(results)
