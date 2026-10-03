const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../retrying');
test('documents an important edge case',()=>{ const { delayFor }=api; assert.equal(delayFor(3,1,20,.25,1),5) });
