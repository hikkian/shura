const test=require('node:test'),assert=require('node:assert/strict'); const { deepMerge }=require('./config-merge');
test('reported behavior',()=>{ const base={x:{y:1}},out=deepMerge(base,{});out.x.y=2;assert.equal(base.x.y,1) });
