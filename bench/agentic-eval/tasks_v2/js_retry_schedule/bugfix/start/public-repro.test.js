const test=require('node:test'),assert=require('node:assert/strict'); const { delayFor }=require('./retrying');
test('reported behavior',()=>{ assert.equal(delayFor(3,1,20,.25,1),5) });
