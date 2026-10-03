import unittest
from routes import match_route
class Routes(unittest.TestCase):
 def test_literals_and_parameters(self):
  self.assertEqual(match_route('/users/:id/files/:name','/users/42/files/a%20b'), {'id':'42','name':'a b'})
  self.assertEqual(match_route('/','/'), {})
  self.assertIsNone(match_route('/users/:id','/users/1/edit'))
  self.assertIsNone(match_route('/users/:id','/users/%FF'))
 def test_invalid_patterns(self):
  for p in ['/x/:/y','/:id/:id']:
   with self.assertRaises(ValueError): match_route(p,'/x/y')
