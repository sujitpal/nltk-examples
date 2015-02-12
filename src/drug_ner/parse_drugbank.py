# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import drug_ner_utils as dnu
import os
import xml.sax

class DrugXmlContentHandler(xml.sax.ContentHandler):
    
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
        self.tags = []
        self.generic_names = []
        self.brand_names = []
        
    def startElement(self, name, attrs):
        self.tags.append(name)
    
    def endElement(self, name):
        self.tags.pop()
        
    def characters(self, content):
        breadcrumb = "/".join(self.tags)
        if breadcrumb == "drugbank/drug/brands/brand":
            self.brand_names.append(content)
        if breadcrumb == "drugbank/drug/name":
            self.generic_names.append(content)
    
def write_list_to_file(lst, filename):
    fout = open(os.path.join(dnu.DATA_DIR, filename), 'wb')
    for e in lst:
        fout.write("%s\n" % (e.encode("utf-8")))
    fout.close()

    
source = open(os.path.join(dnu.DATA_DIR, "drugbank.xml"), 'rb')
handler = DrugXmlContentHandler()
xml.sax.parse(source, handler)
source.close()

write_list_to_file(handler.generic_names, "generic_names.txt")
write_list_to_file(handler.brand_names, "brand_names.txt")

generic_fd = dnu.ngram_distrib(handler.generic_names, dnu.GRAM_SIZE)
brand_fd = dnu.ngram_distrib(handler.brand_names, dnu.GRAM_SIZE)

joblib.dump(generic_fd, os.path.join(dnu.DATA_DIR, "generic_fd.pkl"))
joblib.dump(brand_fd, os.path.join(dnu.DATA_DIR, "brand_fd.pkl"))

# Plot visualizations
dnu.plot_ngram_distrib(generic_fd, 30, "Generic", dnu.GRAM_SIZE)
dnu.plot_ngram_distrib(brand_fd, 30, "Brand", dnu.GRAM_SIZE)

