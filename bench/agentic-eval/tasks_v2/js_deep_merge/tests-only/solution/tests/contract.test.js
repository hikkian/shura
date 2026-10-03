const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../config-merge');
test('documents an important edge case',()=>{ const { deepMerge }=api; const base={x:{y:1}},out=deepMerge(base,{});out.x.y=2;assert.equal(base.x.y,1) });
