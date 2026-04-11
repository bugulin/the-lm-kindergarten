arguments:
-path to json file with syllogisms
-path for the output file
-(optional) path for the summary file

the 4 output categories:
-valid: the syllogism is (likely) of the correct form
-uncertain: the syllogism is likely of the correct form, however, it was probably written with some more exotic language so the code could not match it confidently to the pattern. I would include these to the train set.
-uncertain_count: there were more than 3 terms identified. In some cases, it is caused by structure matching and the syllogisms are in fact of the correct form. However, in some cases, the syllogisms are indeed of incorrect form. Since it seems to be cca 50/50, I would be very careful with these.
-incorrect: exclude these.

Notes:
There are still some important cases worth adding, however, I want to commit at least this version since it is a lot better than the previous one.