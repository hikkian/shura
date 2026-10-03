const test=require('node:test'), assert=require('node:assert/strict'); const {mergePages}=require('../pages');
test('deduplicates first occurrence and sorts',()=>{ const a=[{id:3,v:'first'},{id:1},{id:2}], b=[{id:3,v:'later'},{id:2,v:'later'},{id:4}]; assert.deepEqual(mergePages([a,b],'id'),[{id:1},{id:2},{id:3,v:'first'},{id:4}]); assert.equal(a[0].v,'first'); });
test('empty and absent field',()=>{assert.deepEqual(mergePages([],'id'),[]); assert.throws(()=>mergePages([[{}]],'id'));});

test('public package is split',()=>{const fs=require('node:fs');assert.ok(fs.existsSync('pages/index.js'));assert.ok(fs.existsSync('pages/core.js'));assert.ok(!fs.existsSync('pages.js'));});
