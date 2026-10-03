const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../deps');
test('documents an important edge case',()=>{ const { plan }=api; assert.deepEqual(plan({app:['lib','util'],lib:['core'],util:['core'],core:[],z:['ghost']}),['core','ghost','lib','util','app','z']) });
