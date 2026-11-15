import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench/agentic-eval/note'))
import note
class NoteTests(unittest.TestCase):
 def test_lesson_literal_params(self):
  f={'params':{'x':'test_foo.py'}}
  self.assertTrue(note.check_lesson_text('run test_foo.py',f));self.assertFalse(note.check_lesson_text('run tests',f))
 def test_classify(self):
  self.assertEqual(note.classify('pytest tests/test_foo.py',{**note.KINDS[0], 'params': {'test_file':'tests/test_foo.py'}}),'accepted')
 def test_builders(self):
  p={'body':{'messages':[{'role':'system','content':'s'}]}}
  for m in note.MODES:
   b=note.make_body(p,m,'lesson',['d'],2);self.assertIn('messages',b)
 def test_analysis_insufficient(self):
  self.assertEqual(note.analyze([])['verdict'],'недостаточно данных')
 def test_analysis_known(self):
  rows=[]
  for i in range(48):
   for mode,cl in [('base','old mistake'),('base2','old mistake'),('near','accepted'),('pinned20','accepted'),('pinned200','accepted'),('long_near','accepted')]: rows.append({'pair_id':str(i),'mode':mode,'class':cl,'truncated':False,'idea':str(i%8)})
  self.assertEqual(note.analyze(rows)['verdict'],'заметка держится')
if __name__=='__main__':unittest.main()
