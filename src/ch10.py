#!/usr/bin/python
# Analyzing meaning of sentences

from __future__ import division
import nltk
import re

def english_to_sql():
  nltk.data.show_cfg("grammars/book_grammars/sql0.fcfg")
  from nltk import load_parser
  cp = load_parser("grammars/book_grammars/sql0.fcfg", trace=3)
  query = "What cities are located in China"
  trees = cp.nbest_parse(query.split())
  answer = trees[0].node['SEM']
  q = " ".join(answer)
  print q
  from nltk.sem import chat80
  rows = chat80.sql_query('corpora/city_database/city.db', q)
  for r in rows:
    print r[0],

def logic_parser():
  lp = nltk.LogicParser()
  SnF = lp.parse('SnF')
  NotFnS = lp.parse('-FnS')
  R = lp.parse('SnF -> -FnS')
#  prover = nltk.Prover9()
#  print prover.prove(NotFnS, [SnF, R])
  val = nltk.Valuation([('P',True), ('Q', True), ('R', False)])
  dom = set([])
  g = nltk.Assignment(dom)
  m = nltk.Model(dom, val)
  print "eval(P&Q)=", m.evaluate('(P & Q)', g)
  print "eval -(P&Q)=", m.evaluate('-(P & Q)', g)
  print "eval(P&R)=", m.evaluate('(P & R)', g)
  print "eval(-(P|R))=", m.evaluate('-(P | R)', g)

def first_order_logic():
  tlp = nltk.LogicParser(type_check=True)
  sig = {"walk": "<e,t>"}
  parsed = tlp.parse("walk(angus)", sig)
  print "parsed_arg(value,type)=", parsed.argument, parsed.argument.type
  print "parsed_func(value,type)=", parsed.function, parsed.function.type

def truth_model():
  domain = set(['b', 'o', 'c'])
  v = """
  bertie => b
  olive => o
  cyril => c
  boy => {b}
  girl => {o}
  dog => {c}
  walk => {o, c}
  see => {(b,o), (c,b), (o,c)}
  """
  val = nltk.parse_valuation(v)
  print val
  print ('o', 'c') in val["see"]
  print ('b',) in val["boy"]
  g = nltk.Assignment(domain, [('x', 'o'), ('y', 'c')])
  model = nltk.Model(domain, val)
  print "model.evaluate=", model.evaluate("see(olive,y)", g)
  
def main():
#  english_to_sql()
#  logic_parser()
#  first_order_logic()
  truth_model()
  print "end"
  
if __name__ == "__main__":
  main()
