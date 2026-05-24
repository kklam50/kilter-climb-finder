## General Project Flow
1. Export all climbs and prune excess data
2. tokenize everything
3. local llm setup
4. training
5. probably vibe code a fronend


## Relevant tables
climb_stats
- climb metadata
- round display_difficulty to nearest int when applying the letter grade
climbs: SELECT * FROM climbs WHERE angle IS NOT NULL AND layout_id = 1;
- climb name, uuid, layout (frames)
- layout example (Second Chance): p1099r15p1125r12p1163r12p1214r13p1232r13p1263r13p1316r13p1351r13p1371r13p1390r14p1466r15p1528r15
difficulty_grades
- numerical to vgrade mapping
placements
- maps the layout string to the holes value (ex. p1099: 1099 is the id for placements; placements table contains hole_id)
holes
- mapping of hold to xy coordinate position

## Training 
- probably train on the patterns themselves first, associate the pattern with the uuid of the set (technically already done with the climbs table)
- most important parts: 
    - climb name (climbs table)
    - climb uuid (climbs table)
    - layout as a grid (need to rerepresent the frames column values as some pattern)
        - will require referencing placements, holes tables
        - framePValue (string in climbs table) -> placements -> holes (to get the )