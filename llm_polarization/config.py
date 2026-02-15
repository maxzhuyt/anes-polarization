"""
Configuration for LLM polarization analysis.

This module contains all topic definitions, model paths, and other settings.
"""

# ==========================================
# TOPICS DICTIONARY
# Maps topic keys to their natural language descriptions
# ==========================================
TOPICS = {
    "Baseline": None,

    # =========================
    # IMMIGRATION
    # =========================
    "imm_unauth": "unauthorized and undocumented immigrants",
    "birthright": "birthright citizenship",
    "illeg_child": "children of undocumented immigrants",
    "border_wall": "a wall along the U.S.–Mexico border",
    "spend_border": "whether to increase or decrease funding for border security",
    "immig_levels": "the level of immigration into the United States",
    "biden_immigration": "Joe Biden's handling of immigration",

    # =========================
    # LABOR / FAMILY POLICY
    # =========================
    "paid_leave": "paid parental or family leave",
    "job_gov_guar": "whether the government should guarantee a job to everyone",
    "min_wage": "whether to change the minimum wage",
    "ft_union": "labor unions",

    # =========================
    # INSTITUTIONS / ACCOUNTABILITY / MEDIA
    # =========================
    "trump_corr": "whether Donald Trump was involved in political corruption",
    "journ_access": "whether journalists should have broad access to government officials and information",
    "checks_power": "to what degree the branches of government should limit each other's power",
    "rus_interf": "how serious a problem Russian interference in U.S. elections is",
    "ft_sci": "scientists",

    # =========================
    # ELECTIONS / DEMOCRACY
    # =========================
    "voter_id": "whether voters should be required to show identification to vote",
    "felon_vote": "whether people with felony convictions should have the right to vote",
    "vote_denied": "how often people are denied the right to vote",

    # =========================
    # ECONOMY (RETROSPECTIVE)
    # =========================
    "econ_now": "how good or bad the current national economy is",
    "biden_economy": "Joe Biden's handling of the economy",

    # =========================
    # GUNS
    # =========================
    "gun_bkg_chk": "whether background checks should be required for gun purchases",
    "ar_ban": "whether assault weapons should be banned",
    "gun_imp": "to what degree gun regulation is an important political issue",

    # =========================
    # CRIME / POLICING / ORDER
    # =========================
    "death_pen": "whether the death penalty should be used for serious crimes",
    "police_force": "to what degree police should be allowed to use force",
    "urban_unrest": "how serious a problem urban unrest and protests are",
    "ft_police": "the police",
    "biden_crime": "Joe Biden's handling of crime",
    "crime_spend": "to what degree the government should spend money on dealing with crime",

    # =========================
    # ABORTION / COURTS
    # =========================
    "abortion": "whether abortion should be legal",
    "scotus_abort": "the Supreme Court decisions related to abortion",
    "biden_abortion": "Joe Biden's handling of abortion",

    # =========================
    # HEALTH
    # =========================
    "govt_health": "whether the government should provide health insurance",
    "obamacare": "whether the Affordable Care Act should be kept, expanded, or repealed",
    "vax_school": "whether children should be required to be vaccinated to attend school",
    "health_spend": "to what degree the government should spend money to help people pay for health insurance",

    # =========================
    # EDUCATION / SPENDING
    # =========================
    "spend_school": "to what degree the government should spend money on public schools",
    "dei_college": "diversity, equity, and inclusion (DEI) policies on college campuses",
    "affirm_action": "affirmative action in university admissions",

    # =========================
    # REDISTRIBUTION / WELFARE / TAX
    # =========================
    "spend_welfare": "to what degree the government should spend money on welfare programs",
    "spend_poor": "to what degree the government should spend money to help the poor",
    "svc_spend": "to what degree the government should spend money on public services",
    "millionaire_tax": "whether to tax on millionaires",

    # =========================
    # DIVERSITY (formerly Race)
    # =========================
    "assist_black": "to what degree the government should help Black Americans",
    "black_favor": "whether Black people should get special favor",
    "diversity": "to what degree diversity benefits the country",
    "ft_asian": "Asian-Americans",
    "discuss_race": "how often racial issues should be discussed with children",

    # =========================
    # TRANSGENDER
    # =========================
    "trans_bath": "whether transgender people should use bathrooms matching their gender identity",
    "trans_military": "whether transgender people should serve in the U.S. military",
    "ft_trans": "transgender people",

    # =========================
    # LESBIAN / GAY
    # =========================
    "lg_job": "protection for gay and lesbian people from job discrimination",
    "lg_marry": "legal marriage for same-sex couples",
    "lg_refuse_service": "whether businesses should be allowed to refuse service to same-sex couples",

    # =========================
    # GENDER
    # =========================
    "ft_fem": "feminists",

    # =========================
    # CLIMATE / ENVIRONMENT
    # =========================
    "clim_imp": "how important climate change is as an issue",
    "env_bus": "the tradeoff between environmental protection and business interests",
    "ghg_emiss": "whether to regulate greenhouse gas emissions",

    # =========================
    # DEFENSE / FOREIGN POLICY
    # =========================
    "def_spend": "to what degree the government should spend money on national defense",
    "mil_force": "whether the United States should use military force in foreign countries",
    "biden_foreign": "Joe Biden's handling of foreign relations",
    "israel_aid": "U.S. military assistance to Israel",

    # =========================
    # TRADE
    # =========================
    "free_trade": "free trade agreements with other countries",
    "intl_trade_job": "whether international trade helps or hurts jobs in the United States",
    "limit_imports": "placing new limits on imports",
}

TOPICS_GSS  = {
    # =========================
    # NATIONAL SPENDING (nat...)
    # =========================
    "natspac": "whether to spend more money on the space exploration program",
    "natspacy": "whether to spend more money on space exploration (version y)",
    "natenvir": "whether to spend more money on protecting the environment",
    "natenviy": "whether to spend more money on the environment (version y)",
    "natheal": "whether to spend more money on the nation's health",
    "nathealy": "whether to spend more money on health (version y)",
    "natcity": "whether to spend more money on solving big city problems",
    "natcityy": "whether to spend more money on assistance to big cities (version y)",
    "natcrime": "whether to spend more money on halting the crime rate",
    "natcrimy": "whether to spend more money on law enforcement (version y)",
    "natdrug": "whether to spend more money on dealing with drug addiction",
    "natdrugy": "whether to spend more money on drug rehabilitation (version y)",
    "nateduc": "whether to spend more money on the nation's education system",
    "nateducy": "whether to spend more money on education (version y)",
    "natrace": "whether to spend more money on improving conditions for Black people",
    "natracey": "whether to spend more money on assistance to Black people (version y)",
    "natarms": "whether to spend more money on military and defense",
    "natarmsy": "whether to spend more money on national defense (version y)",
    "nataid": "whether to spend more money on foreign aid",
    "nataidy": "whether to spend more money on assistance to other countries (version y)",
    "natfare": "whether to spend more money on welfare",
    "natfarey": "whether to spend more money on assistance to the poor (version y)",
    "natroad": "whether to spend more money on highways and bridges",
    "natsoc": "whether to spend more money on the social security system",
    "natmass": "whether to spend more money on mass transportation",
    "natpark": "whether to spend more money on parks and recreation",
    "natsci": "whether to spend more money on scientific research",
    "natenrgy": "whether to spend more money on alternative energy sources",
    "natchld": "whether to spend more money on government childcare assistance",

    # =========================
    # IDEOLOGY & GOVERNMENT ROLE
    # =========================
    "polviews": "where to fall on the spectrum from liberal to conservative",
    "govchrst": "whether the federal government should advocate Christian values",
    "goveqinc": "whether the government should act to reduce income differentials",
    "goveqinc1": "the degree of government responsibility to fix income differences",
    "govfnaid": "the degree of support for government financial aid to low-income students",
    "govfnanc": "the degree of support for government job-creation projects",
    "govunemp": "whether the government should provide unemployment benefits",
    "govineq1": "whether politicians care about reducing income differences",
    "govineq2": "how successful the U.S. government is in reducing income differences",
    "ldctax": "whether rich countries should pay taxes to help poor countries",
    "hlthgov": "whether the government should provide only limited health care",
    "eqwlth": "the degree to which the government should reduce income differences",
    "helppoor": "whether the government should improve the standard of living",
    "helpnot": "whether the government should do more or less to solve problems",
    "helpsick": "whether the government should help pay for medical care",

    # =========================
    # REPRODUCTIVE & BIOETHICS
    # =========================
    "abany": "whether to allow legal abortion for any reason",
    "abdefect": "whether to allow abortion if there is a chance of serious birth defects",
    "abnomore": "whether to allow abortion for married women who want no more children",
    "abhlth": "whether to allow abortion if health is endangered",
    "abpoor": "whether to allow abortion if a low-income woman cannot afford more children",
    "abrape": "whether to allow abortion if the pregnancy resulted from rape",
    "absingle": "whether to allow abortion for women who are not married",
    "abhelp1": "willingness to help with arrangements for an abortion",
    "abhelp2": "willingness to help pay for an abortion",
    "abhelp3": "willingness to help with other abortion-related costs",
    "abhelp4": "willingness to provide emotional support for an abortion",
    "pillok": "the degree of support for providing birth control to teenagers",
    "letdie1": "whether to allow doctors to end the lives of incurable patients",
    "letdie1y": "whether to allow incurable patients to die (version y)",

    # =========================
    # ENVIRONMENT (NON-SPENDING)
    # =========================
    "carsgen": "the degree of danger car pollution poses to the environment",
    "grncon": "the degree of concern about the environment",
    "grnecon": "whether too much worry is placed on the environment versus the economy",
    "grnexagg": "whether environmental threats are exaggerated",
    "grnprog": "whether progress harms the environment",
    "grnprice": "willingness to pay higher prices to help the environment",
    "grnsol": "willingness to accept a lower living standard to help the environment",
    "grntaxes": "willingness to pay higher taxes to help the environment",
    "grwtharm": "whether economic growth always harms the environment",
    "grwthelp": "whether economic growth is necessary to protect the environment",
    "harmsgrn": "whether almost everything people do harms the environment",
    "ihlpgrn": "taking personal action to help the environment",
    "impgrn": "the degree of importance placed on the environment relative to other things",
    "toodifme": "how difficult it is for individuals to help the environment",
    "othssame": "refusing to save the environment unless others do the same",

    # =========================
    # JUSTICE & POLICE
    # =========================
    "cappun": "whether to support the death penalty for murder",
    "gunlaw": "whether to favor or oppose gun permits",
    "courts": "whether courts are too harsh or too lenient",
    "grass": "whether to legalize marijuana",
    "polhitok": "whether police can strike a citizen",
    "polabuse": "whether police can strike a citizen who used vulgar language",
    "polescap": "whether police can strike a citizen attempting to escape",
    "polmurdr": "whether police can strike a murder suspect",
    "polattak": "whether police can strike a citizen attacking with fists",

    # =========================
    # RACE & EQUITY
    # =========================
    "racdif1": "whether racial differences are due to discrimination",
    "racdif2": "whether racial differences are due to in-born ability",
    "racdif3": "whether racial differences are due to lack of education",
    "racdif4": "whether racial differences are due to lack of will",
    "helpblk": "whether the government should aid Black people",
    "wrkwayup": "whether Black people should overcome prejudice without special favors",
    "affrmact": "the degree of support for preference in hiring Black people",
    "discaff": "whether white people are hurt by affirmative action",
    "wlthwhts": "the perceived degree of wealth differences between races",
    "workwhts": "the perceived degree of work ethic differences between races",
    "intlwhts": "the perceived degree of intelligence differences between races",
    "racopen": "whether to support an open housing law",
    "marasian": "whether to favor a close relative marrying an Asian person",
    "marwht": "whether to favor a close relative marrying a white person",
    "marblk": "whether to favor a close relative marrying a Black person",
    "marhisp": "whether to favor a close relative marrying a Hispanic person",

    # =========================
    # INSTITUTIONAL CONFIDENCE
    # =========================
    "confed": "the level of confidence in the executive branch of the federal government",
    "conjudge": "the level of confidence in the United States Supreme Court",
    "conlegis": "the level of confidence in Congress",
    "conbus": "the level of confidence in major companies",
    "confinan": "the level of confidence in banks and financial institutions",
    "conarmy": "the level of confidence in the military",
    "consci": "the level of confidence in the scientific community",
    "scienthe": "the degree to which scientists help solve problems",
    "coneduc": "the level of confidence in the education system",
    "conclerg": "the level of confidence in organized religion",

    # =========================
    # CIVIL LIBERTIES & TOLERANCE
    # =========================
    "spkath": "whether to allow an anti-religionist to speak",
    "colath": "whether to allow an anti-religionist to teach",
    "libath": "whether to allow anti-religious books in the library",
    "spkcom": "whether to allow a communist to speak",
    "colcom": "whether a communist teacher should be fired",
    "libcom": "whether to allow a communist's book in the library",
    "spkrac": "whether to allow a racist to speak",
    "colrac": "whether to allow a racist to teach",
    "librac": "whether to allow a racist's book in the library",
    "spkhomo": "whether to allow a homosexual to speak",
    "colhomo": "whether to allow a homosexual to teach",
    "libhomo": "whether to allow a homosexual's book in the library",
    "spkmslm": "whether to allow anti-American Muslim clergymen to speak",
    "colmslm": "whether to allow anti-American Muslim clergymen to teach",
    "libmslm": "whether to allow anti-American Muslim clergymen's books in the library",
    "spkmil": "whether to allow a militarist to speak",
    "colmil": "whether to allow a militarist to teach",
    "libmil": "whether to allow a militarist's book in the library",

    # =========================
    # GENDER & SOCIAL VALUES
    # =========================
    "fejobaff": "the degree of support for preferential hiring of women",
    "fehire": "the degree of support for hiring and promoting women",
    "discaffw": "whether women lose jobs due to discrimination",
    "discaffm": "whether men lose jobs due to discrimination",
    "sexeduc": "whether to support sex education in public schools",
    "prayer": "whether to support bible prayer in public schools",
    "religinf": "whether the U.S. would be better if religion had less influence",
    "spanking": "whether to approve of spanking to discipline children",
    "helpful": "whether people are generally helpful",
    "getahead": "whether hard work or luck determines success",
    "eldfnce": "whether grandparents should help grandchildren financially",
    "uswary": "the perceived likelihood of the U.S. being in a world war"
}
# ==========================================
# 7-POINT IDEOLOGICAL SCALE LABELS
# Used for generic ideological prompts
# ==========================================
IDEOLOGY_LABELS = {
    1: "extremely liberal",
    2: "liberal",
    3: "slightly liberal",
    4: "moderate",
    5: "slightly conservative",
    6: "conservative",
    7: "extremely conservative",
}

# Alternative label sets for experimentation
IDEOLOGY_LABELS_ALT = {
    1: "far left",
    2: "left",
    3: "center-left",
    4: "centrist",
    5: "center-right",
    6: "right",
    7: "far right",
}

# ==========================================
# ANES IDEOLOGY DISTRIBUTION
# From ideology_7pt in ANES 2020+2024 combined
# Used for weighted sampling to match real population
# ==========================================
ANES_IDEOLOGY_COUNTS = {
    1: 619,   # extremely liberal
    2: 2029,  # liberal
    3: 1461,  # slightly liberal
    4: 3005,  # moderate
    5: 1415,  # slightly conservative
    6: 2522,  # conservative
    7: 737,   # extremely conservative
}

# Pre-computed proportions (excluding moderates for lib/con comparison)
ANES_IDEOLOGY_PROPORTIONS = {
    1: 0.052511,
    2: 0.172124,
    3: 0.123940,
    4: 0.254920,  # excluded in metrics
    5: 0.120037,
    6: 0.213946,
    7: 0.062521,
}

# ==========================================
# MODEL CONFIGURATION
# ==========================================
DEFAULT_MODEL_PATH = "/project/jevans/maxzhuyt/models/Meta-Llama-3.1-8B-Instruct"
DEFAULT_BATCH_SIZE = 128

# ==========================================
# DATA PATHS
# ==========================================
NOMINATE_CSV = "/project/jevans/maxzhuyt/data/HS116_members_fullname.csv"
ANES_POLARIZATION_CSV = "policy_polarization.csv"

# ==========================================
# SYSTEM MESSAGES FOR PROMPTING
# ==========================================
SYSTEM_MSG_POLITICIAN = (
    "You are simulating the public stance of U.S. politicians.\n\n"
)

SYSTEM_MSG_IDEOLOGY = (
    "You are simulating the views of an American.\n\n"
)
