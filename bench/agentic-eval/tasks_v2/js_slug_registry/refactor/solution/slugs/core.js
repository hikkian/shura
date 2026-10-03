function slugify(text) {
  if (typeof text !== 'string') throw new TypeError('text must be a string');
  const value = text.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  return value || 'item';
}
function uniqueSlug(text, used) {
  const base = slugify(text); if (!used.has(base)) return base;
  let n = 2; while (used.has(`${base}-${n}`)) n++; return `${base}-${n}`;
}
module.exports = { slugify, uniqueSlug };
