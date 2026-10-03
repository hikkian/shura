const test=require('node:test'),assert=require('node:assert/strict'); const { plan }=require('./deps');
test('reported behavior',()=>{ assert.deepEqual(plan({app:['lib','util'],lib:['core'],util:['core'],core:[],z:['ghost']}),['core','ghost','lib','util','app','z']) });
