const test=require('node:test'),assert=require('node:assert/strict'); const { replay }=require('./journal');
test('reported behavior',()=>{ const events=[{seq:1,op:'set',key:'x',value:2},{seq:1,op:'set',key:'x',value:9},{seq:2,op:'add',key:'x',value:3}];assert.deepEqual(replay(events),{x:5}) });
