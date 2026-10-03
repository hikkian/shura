const test=require('node:test'),assert=require('node:assert/strict'); const { slugify, uniqueSlug }=require('./slugs');
test('reported behavior',()=>{ assert.equal(slugify('A__B'), 'a-b') });
