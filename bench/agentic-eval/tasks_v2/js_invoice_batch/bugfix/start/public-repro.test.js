const test=require('node:test'),assert=require('node:assert/strict'); const { invoice }=require('./invoice');
test('reported behavior',()=>{ assert.equal(invoice([{quantity:1,unitCents:5}],1000).tax,1) });
