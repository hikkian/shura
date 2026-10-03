const test=require('node:test'), assert=require('node:assert/strict');
const api=require('../slugs');
test('documents an important edge case',()=>{ const { slugify, uniqueSlug }=api; assert.equal(slugify('A__B'), 'a-b') });
