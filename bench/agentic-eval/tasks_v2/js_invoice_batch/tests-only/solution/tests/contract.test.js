const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../invoice');
test('documents an important edge case',()=>{ const { invoice }=api; assert.equal(invoice([{quantity:1,unitCents:5}],1000).tax,1) });
