const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../routes');
test('documents an important edge case',()=>{ const { matchRoute }=api; assert.equal(matchRoute('/users/:id','/users/1/edit'),null) });
