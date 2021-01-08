from os.path import join, dirname

SOURCE_DIR = "../dataset"

HANS_SOURCE = join(SOURCE_DIR, "hans")

GLUE_SOURCE = join(SOURCE_DIR, "glue_multinli")
MULTINLI_SOURCE = join(SOURCE_DIR, "multinli")
MNLI_TEST_SOURCE = join(SOURCE_DIR, "multinli_test")
MNLI_HARD_SOURCE = join(SOURCE_DIR, "multinli_hard")

TEACHER_SOURCE = join("../teacher_preds", "mnli_teacher_probs.json")
QQP_TEACHER_SOURCE = join("../teacher_preds", "qqp_teacher_seed222.json")

MNLI_WORD_OVERLAP_BIAS = "../biased_preds/lex_overlap_preds.pkl"
MNLI_WORD_OVERLAP_BIAS_DEV = "../biased_preds/mnli_dev_lex_overlap_preds.pkl"
MNLI_WORD_OVERLAP_BIAS_HANS = "../biased_preds/hans_lex_overlap_preds.pkl"

LEX_BIAS_SOURCE = join("../biased_preds", "lex_overlap_preds.json")
HYPO_BIAS_SOURCE = join("../biased_preds", "hyp_only.json")

BIAS_PATHS = [("hypo", "hyp_only.json"), ("hans_json", "lex_overlap_preds.json"),
              ("dam", "dam_preds.json"), ("qqp_hans_json", "qqp_hans_preds.json"),
              ("fever_claim_only", "duplicate_fever_claim_only.json"),
              ("fever_claim_only_balanced", "weaker_balanced_duplicate_fever_claim_only.json"),
              ("fever_claim_only_bow", "bow_fever_claim_only.json"),
              ("fever_claim_only_bow_reproduce", "bow_reproduce_fever_claim_only.json"),
              ("fever_claim_only_infersent", "infersent_fever_claim_only.json")]
BIAS_SOURCES = {x[0]: join("../biased_preds", x[1]) for x in BIAS_PATHS}

###########################

QQP_SOURCE = join(SOURCE_DIR, "QQP")
QQP_ADD_PAWS = join(SOURCE_DIR, "qqp_paws")
QQP_PAWS_SOURCE = join(SOURCE_DIR, "paws", "qqp")

QQP_PAWS_PATHS = [("hans_0", "hans_0.json"), ("hans_100", "hans_100.json"), ("hans_250", "hans_250.json"),
                  ("hans_500", "hans_500.json"), ("hans_1000", "hans_1000.json"), ("hans_1500", "hans_1500.json"),
                  ("hans_2000", "hans_2000.json"), ("hans_2500", "hans_2500.json")]
QQP_PAWS_PATHS += [("shallow_0", "qqp_shallow_0.json")]

QQP_PAWS_BIAS_SOURCES = {x[0]: join("../biased_preds/qqp_paws", x[1]) for x in QQP_PAWS_PATHS}
QQP_PAWS_TEACHER_SOURCES = {x[0]: join("../teacher_preds/qqp_paws", x[1]) for x in QQP_PAWS_PATHS}

###########################
FEVER_SOURCE = join(SOURCE_DIR, "fever")
FEVER_TEACHER_SOURCE = join("../teacher_preds", "fever_teacher.json")