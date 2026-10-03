const test = require('node:test'); const assert = require('node:assert/strict'); const {slugify, uniqueSlug} = require('../slugs');
test('normalizes unicode and separators', () => { assert.equal(slugify('Crème brûlée / v2'), 'creme-brulee-v2'); assert.equal(slugify('---'), 'item'); assert.equal(slugify('A__B'), 'a-b'); });
test('chooses first free suffix', () => { assert.equal(uniqueSlug('Hello world', new Set()), 'hello-world'); assert.equal(uniqueSlug('Hello world', new Set(['hello-world','hello-world-2','hello-world-4'])), 'hello-world-3'); });
test('rejects wrong input', () => assert.throws(() => slugify(null), TypeError));

test('public package is split',()=>{const fs=require('node:fs');assert.ok(fs.existsSync('slugs/index.js'));assert.ok(fs.existsSync('slugs/core.js'));assert.ok(!fs.existsSync('slugs.js'));});
