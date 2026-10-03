const test=require('node:test'),assert=require('node:assert/strict'); const { enabled }=require('./rollout');
test('reported behavior',()=>{ assert.equal(enabled('new-ui','user-17',37,'release'),true) });
