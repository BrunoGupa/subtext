-- How often does each "peninsular" word actually occur in the Mexican corpus?
--
-- Run this after any corpus rebuild, before touching NOT_MEXICAN in localise.py.
-- The detector's 66-word list is a bulk statistical signal over 1,000-line windows; it is
-- NOT a rule about a single line. 59 of the 66 appear in Mexican Spanish. Only forms at
-- the very bottom of this output belong in a hard reject list.
SELECT
  word,
  mx AS occurrences_in_mx_corpus,
  round(mx / (SELECT count() FROM subtext.mx_corpus) * 1e6, 1) AS per_million_mx
FROM (
  SELECT arrayJoin([
    'vale','tio','coche','piso','tia','joder','jodido','cono','vosotros','coches',
    'movil','dormitorio','billete','alquiler','vales','ascensor','jodida','gafas',
    'pisos','billetes','tios','vuestro','cazadora','chaval','guay','gilipollas',
    'ordenador','currar','curro','flipar','mola','cutre','chorrada','mogollon',
    'aparcar','majo','pijo','cabrear','fontanero','pajita','chavales'
  ]) AS word
) AS w
LEFT JOIN (
  SELECT word, count() AS mx FROM (
    SELECT arrayJoin(splitByChar(' ',
      concat(' ', replaceRegexpAll(
        translateUTF8(lowerUTF8(es), 'áéíóúñü', 'aeiounu'), '[^a-z0-9]+', ' '), ' '))) AS word
    FROM subtext.mx_corpus
  ) WHERE word != '' GROUP BY word
) AS c USING (word)
ORDER BY mx ASC;
