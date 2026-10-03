const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../rollout');
test('documents an important edge case',()=>{ const { enabled }=api; assert.equal(enabled('new-ui','user-17',37,'release'),true) });
