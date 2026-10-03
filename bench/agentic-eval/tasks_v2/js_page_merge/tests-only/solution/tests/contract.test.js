const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../pages');
test('documents an important edge case',()=>{ const { mergePages }=api; assert.deepEqual(mergePages([[{id:3,v:'first'},{id:1}],[{id:3,v:'later'},{id:2}]],'id'),[{id:1},{id:2},{id:3,v:'first'}]) });
