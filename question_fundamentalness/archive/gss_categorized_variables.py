# GSS Variables Categorized into Public Issues and Private Life
# FILTERED VERSION: Removed factual/behavioral questions and non-ordered answer questions
#
# Category Definitions:
# 1. PUBLIC_ISSUES: Opinions about policy, government, social issues connecting to policy
#    (abortion laws, government spending, race relations, environment policy, free speech,
#    affirmative action, welfare, foreign policy, science policy, criminal justice, etc.)
#
# 2. PRIVATE_LIFE: Everyday habits, personal choices, leisure, beliefs about life/morality/values
#    that either don't relate immediately to concrete policy OR describe everyday habits
#    (horoscope reading, religious attendance, personal relationships, museum visits,
#    beliefs about premarital sex, happiness, personal values, etc.)
#
# Filtering criteria applied:
# - Removed factual questions (e.g., "Are you a member of...", "Do you have a gun...")
# - Removed non-ordered answer questions (e.g., "which is most important", categorical selections)

PUBLIC_ISSUES = [
    # Abortion policy
    "abany",           # abortion if woman wants for any reason
    "abdefect",        # strong chance of serious defect
    "abhlth",          # woman's health seriously endangered
    "abnomore",        # married--wants no more children
    "abpoor",          # low income--cant afford more children
    "abrape",          # pregnant as result of rape
    "absingle",        # not married

    # Abortion-related personal actions (policy-connected)
    "abhelp1",         # r would help with arrangements for abortion
    "abhelp2",         # r would help with paying for abortion
    "abhelp3",         # r would help with paying for other costs
    "abhelp4",         # r would help with emotional support

    # Science/research funding and attitudes toward scientists
    "advfront",        # sci rsch should be supported by federal govt
    "balneg",          # sci research harmful results outweigh benefits
    "balpos",          # sci research benefits outweigh harmful
    "scibnfts",        # benefits of sci research outweigh harmful results
    "scientbe",        # scientists want to make life better for avg person
    "scientgo",        # scientists work for good of humanity
    "scienthe",        # scientists help solve problems
    "scientod",        # scientists are odd and peculiar
    "nanowill",        # benefit of nanotechnology outweigh harmful results

    # Affirmative action and discrimination
    "affrmact",        # Favor preference in hiring Black people
    "discaff",         # whites hurt by aff. action
    "discaffm",        # a man won't get a job or promotion
    "discaffw",        # a woman won't get a job or promotion
    "fehire",          # should hire and promote women
    "fejobaff",        # for or against preferential hiring of women
    "wrkwayup",        # Black people overcome prejudice without favors

    # Environment policy (opinion scales only)
    "carsgen",         # car pollution danger to envir
    "chemgen",         # pesticides danger to envir
    "genegen",         # how dangerous modifying genes in crops
    "grncon",          # concerned about environment (1-5 scale)
    "grnecon",         # worry too much about envir, too little econ
    "grneffme",        # environment effect everyday life
    "grnexagg",        # environmental threats exaggerated
    "grnprice",        # pay higher prices to help envir
    "grnprog",         # worry too much about progress harming envir
    "grnsol",          # accept cut in living stnds to help envir
    "grntaxes",        # pay higher taxes to help envir
    "grwtharm",        # econ grwth always harms envir
    "grwthelp",        # amer needs econ grwth to protect envir
    "harmsgrn",        # almost everything we do harms envir
    "helpharm",        # hard to know if lifestyle helps environment
    "ihlpgrn",         # do what i can to help envir
    "impgrn",          # more important in life than save environment
    "indusgen",        # indust air pollution danger to envir
    "nobuygrn",        # how often avoid products for environment
    "nukegen",         # nuke power danger to envir
    "othssame",        # no saving environment unless others do same
    "scigrn",          # modern science will solve envir probs
    "tempgen",         # greenhouse effect danger to envir
    "tempgen1",        # greenhouse effect danger (climate change)
    "toodifme",        # too diff to do anything about envir
    "watergen",        # water pollution danger to envir

    # Criminal justice and policing
    "cappun",          # favor or oppose death penalty for murder
    "courts",          # courts dealing with criminals
    "polabuse",        # approve police striking citizen who cursed
    "polattak",        # approve police striking attacking citizen
    "polescap",        # approve police striking escaping citizen
    "polhitok",        # ever approve of police striking citizen
    "polmurdr",        # approve police striking murder suspect

    # Free speech - teaching
    "colath",          # allow anti-religionist to teach
    "colcom",          # should communist teacher be fired
    "colhomo",         # allow homosexual to teach
    "colmil",          # allow militarist to teach
    "colmslm",         # allow anti-american muslim clergymen teaching
    "colrac",          # allow racist to teach

    # Confidence in institutions
    "conarmy",         # confidence in military
    "conbus",          # confidence in major companies
    "conclerg",        # confidence in organized religion
    "coneduc",         # confidence in education
    "confed",          # confid. in exec branch of fed govt
    "confinan",        # confid in banks & financial institutions
    "conjudge",        # confid. in united states supreme court
    "conlegis",        # confidence in congress

    # Mental health policy (forced treatment)
    "dangroth",        # x should be hospitalized if dangerous to others
    "dangrslf",        # x should be hospitalized if dangerous to self
    "mustdoc",         # x should be forced to be examined by law
    "musthosp",        # x should be hospitalized by law
    "mustmed",         # x should be forced to take medication by law
    "viggrp",          # willing to have group home in neighborhood

    # Divorce law
    "divlaw",          # divorce laws easier or harder

    # Income inequality and government redistribution
    "eqwlth",          # should govt reduce income differences (1-7 scale)
    "goveqinc",        # govmnt should reduce inc differentials
    "goveqinc1",       # govnmnt should fix income differences
    "govineq1",        # politicians do not care about reducing inc differences
    "govineq2",        # us govt success in reducing inc differences

    # Government programs and intervention
    "govchrst",        # federal government should advocate christian values
    "govfnaid",        # govt should give financial assistance to students
    "govfnanc",        # govt should finance projects to create new jobs
    "govunemp",        # govmnt should provide unemp benefits
    "helpblk",         # Should govt aid Black people?
    "helpnot",         # should govt do more or less?
    "helppoor",        # should govt improve standard of living?
    "helpsick",        # should govt help pay for medical care?
    "hlthgov",         # GOVT SHOULD PROVIDE ONLY LIMITED HEALTH CARE

    # Drug policy
    "grass",           # should marijuana be made legal

    # Guns (opinion only, not ownership)
    "gunlaw",          # favor or oppose gun permits

    # Work and gender roles (policy-connected)
    "hubbywk1",        # men should earn money women keep house

    # Interest in policy issues
    "inteduc",         # interested in local school issues
    "intenvir",        # interested in environmental issues
    "intfarm",         # interested in farm issues
    "intmil",          # interested in military policy
    "intspace",        # interested in space exploration

    # International policy
    "ldctax",          # rich countries pay tax to help poor countries
    "uswary",          # expect u.s. in world war in 10 years

    # End of life policy
    "letdie1",         # allow incurable patients to die
    "letdie1y",        # allow incurable patients to die (form 2)

    # Free speech - library books
    "libath",          # allow anti-religious book in library
    "libathy",         # allow anti-religionist book
    "libcom",          # allow communists book in library
    "libhomo",         # allow homosexuals book in library
    "libmil",          # allow militarists book in library
    "libmslm",         # allow anti-american muslim clergymen's books
    "librac",          # allow racists book in library

    # Government spending priorities (nat* series)
    "nataid",          # foreign aid
    "nataidy",         # assistance to other countries
    "natarms",         # military, armaments, and defense
    "natarmsy",        # national defense
    "natchld",         # assistance for childcare
    "natcity",         # solving problems of big cities
    "natcityy",        # assistance to big cities
    "natcrime",        # halting rising crime rate
    "natcrimy",        # law enforcement
    "natdrug",         # dealing with drug addiction
    "natdrugy",        # drug rehabilitation
    "nateduc",         # improving nations education system
    "nateducy",        # education
    "natenrgy",        # developing alternative energy sources
    "natenvir",        # improving & protecting environment
    "natenviy",        # the environment
    "natfare",         # welfare
    "natfarey",        # assistance to the poor
    "natheal",         # improving & protecting nations health
    "nathealy",        # health
    "natmass",         # mass transportation
    "natpark",         # parks and recreation
    "natrace",         # Improving the conditions of Black people
    "natracey",        # Assistance to Black people
    "natroad",         # highways and bridges
    "natsci",          # supporting scientific research
    "natsoc",          # social security
    "natspac",         # space exploration program
    "natspacy",        # space exploration

    # Science and technology policy
    "nextgen",         # science & tech. give more opportunities to next generation

    # Success and mobility (policy-connected views on race/gender)
    "oprace",          # need to be right race to get ahead
    "opsex",           # need to be right sex to get ahead

    # Political views (ordered scale)
    "polviews",        # think of self as liberal or conservative (1-7 scale)

    # Sex education and birth control policy
    "pillok",          # birth control to teenagers 14-16
    "sexeduc",         # sex education in public schools

    # Religion and government
    "prayer",          # bible prayer in public schools
    "religinf",        # us would be better if religion had less influence

    # Race relations and racial attitudes (policy-connected)
    "racdif1",         # differences due to discrimination
    "racdif2",         # differences due to in-born learning ability
    "racdif3",         # differences due to lack of education
    "racdif4",         # differences due to lack of will
    "racopen",         # vote on open housing law
    "intlwhts",        # unintelligent-intelligent rating of whites
    "wlthwhts",        # rich-poor rating of whites
    "workwhts",        # hard working-lazy rating of whites

    # Free speech - speaking
    "spkath",          # allow anti-religionist to speak
    "spkathy",         # allow anti-religionist to speak y
    "spkcom",          # allow communist to speak
    "spkcomy",         # allow communist to speak y
    "spkhomo",         # allow homosexual to speak
    "spkhomoy",        # allow homosexual to speak y
    "spkmil",          # allow militarist to speak
    "spkmily",         # allow militarist to speak y
    "spkmslm",         # allow muslim clergymen preaching hatred
    "spkmslmy",        # allow muslim clergymen preaching hatred y
    "spkrac",          # allow racist to speak
    "spkracy",         # allow racist to speak y

    # Taxes
    "tax",             # r's federal income tax
]

PRIVATE_LIFE = [
    # Sexual behavior and partners (counts and frequencies)
    "PARTNERS_8822",   # How many sex partner's R had in last year
    "acqntsex",        # r had sex with acquaintance last year
    "evpaidsx",        # ever have sex paid for or being paid since 18
    "evstray",         # have sex other than spouse while married
    "frndsex",         # r had sex with friend last year
    "matesex",         # was 1 of r's partner's spouse or regular
    "nummen",          # number of male sex partner's since 18
    "numwomen",        # number of female sex partner's since 18
    "othersex",        # r had sex with some other last year
    "paidsex",         # r had sex for pay last year
    "partners",        # how many sex partner's r had in last year
    "partnrs5",        # how many sex partner's r had in last 5 years
    "pikupsex",        # r had sex with casual date last year
    "sexfreq",         # frequency of sex during last year

    # Beliefs about astrology and supernatural
    "astrolgy",        # ever read a horoscope or personal astrology report
    "astrosci",        # astrology is scientific (ordered scale)

    # Religious practice and personal beliefs
    "attend",          # how often r attends religious services
    "bible",           # feelings about the bible (ordered belief scale)
    "god",             # r's confidence in the existence of god (ordered scale)
    "godmeans",        # life meaningful because god exists
    "makefrnd",        # religion helps people to make friends
    "popespks",        # pope is infallible
    "postlife",        # belief in life after death
    "pray",            # how often does r pray
    "reborn",          # has r ever had a 'born again' experience
    "relexp",          # have religious experience changed life
    "relidesc",        # how well r's religion describes r
    "relidimp",        # how important is r's religion
    "relidins",        # extent r feels insulted if religion criticized
    "relidwe",         # how often r says 'we' about religion
    "religimp",        # how important is religion
    "relpersn",        # r consider self a religious person
    "savesoul",        # tried to convince others to accept jesus
    "sprtprsn",        # r consider self a spiritual person
    "theism",          # god concerned with human beings personally

    # Work schedule and work-life
    "chngtme",         # how often r allowed change schedule
    "famvswk",         # how often fam life interfere job
    "famwkoff",        # how hard to take time off
    "mustwork",        # mandatory to work extra hours
    "wkvsfam",         # how often job interferes fam life
    "wrkhome",         # how often r works at home

    # Drug use (yes/no only)
    "evcrack",         # r ever use crack cocaine
    "evidu",           # r ever inject drugs

    # Personal philosophy and life meaning
    "egomeans",        # life meaningful only if you provide meaning
    "fatalism",        # people can't change the course of their lives
    "nihilism",        # life serves no purpose

    # Email and internet habits
    "emailhr",         # email hours per week
    "emailmin",        # email minutes per week
    "wwwhr",           # www hours per week
    "wwwmin",          # www minutes per week

    # Condom use
    "condom",          # used condom last time

    # Family attitudes about mental health
    "fammhneg",        # r's family's negative attitudes about mh problems
    "othmhneg",        # r's other acquaintences negative attitudes

    # Fear and safety (personal feeling)
    "fear",            # afraid to walk at night in neighborhood

    # Personal values about helping
    "firstyou",        # should take care of oneself first
    "helpfrds",        # should better off ppl help less well-off friend

    # Beliefs about success and getting ahead (importance scales)
    "opclout",         # need political connections to get ahead
    "opeduc",          # need good education to get ahead
    "ophrdwrk",        # need to work hard to get ahead
    "opknow",          # need to know right people to get ahead
    "oppared",         # need educated parents to get ahead
    "oprelig",         # need to be right religion to get ahead
    "opwlth",          # need wealthy family to get ahead

    # Life outlook and optimism
    "goodlife",        # standard of living of r will improve
    "kidssol",         # r's kids living standard compared to r
    "parsol",          # r's living standard compared to parents

    # Happiness and satisfaction
    "hapcohab",        # happiness of relt with partner
    "hapmar",          # happiness of marriage
    "happy",           # general happiness
    "happy7",          # how happy r is
    "hapunhap",        # happy or unhappy with life today
    "life",            # is life exciting or dull
    "satfam7",         # family satisfaction in general
    "satfin",          # satisfaction with financial situation
    "satjob",          # work satisfaction

    # Trust in people
    "helpful",         # people helpful or looking out for selves
    "helpfulnv",       # people helpful (no volunteered response)
    "helpfulv",        # people helpful (with volunteered response)

    # Leisure and relaxation
    "hrsrelax",        # hours per day r have to relax
    "hunt",            # does r or spouse hunt
    "socbar",          # spend evening at bar
    "socfrend",        # spend evening with friends
    "socommun",        # spend evening with neighbor
    "socrel",          # spend evening with relatives
    "visnhist",        # how often r visited natural history museum
    "vissci",          # how often r visited science museum
    "viszoo",          # how often r visited zoo
    "xmovie",          # seen x-rated movie in last year

    # Sexual morality (no direct current policy)
    "homosex",         # homosexual sex relations
    "homosex1",        # is homosexual sex wrong?
    "premarsx",        # sex before marriage
    "teensex",         # sex before marriage: teens 14-16
    "xmarsex",         # sex with person other than spouse
    "xmarsex1",        # is extramarital sex wrong?

    # Family and marriage attitudes (non-policy)
    "eldfnce",         # GRANDPARENTS SHOULD HELP GRANDCHILDREN FINANCIALLY
    "marasian",        # close relative marry asian
    "marmakid",        # single ma can raise kids well as couple
    "marpakid",        # single pa can raise kids well as couple
    "marwht",          # r favor close relative marrying white person
    "meovrwrk",        # men hurt family when focus on work too much
    "relmarry",        # r accepts person from different religion marrying relative

    # Environmental recycling (everyday habit, not policy view)
    "recycle",         # recycle cans bottles

    # Neighborhood perceptions
    "raclive",         # any opp. race in neighborhood

    # Looking after family (everyday activity)
    "rfamlook",        # hours r spends looking after family members
    "spfalook",        # hours spouse spends looking after family

    # Work attitudes
    "richwork",        # if rich, continue or stop working

    # Spanking (parenting, personal)
    "spanking",        # favor spanking to discipline child

    # Science and religion (personal belief, not policy)
    "toofast",         # science makes our way of life change too fast
    "trustsci",        # we trust too much in science

    # Personal interest in topics (not policy opinions)
    "intmed",          # interested in medical discoveries
    "intsci",          # interested in new scientific discoveries
    "inttech",         # interested in technologies
]

# Print summary
if __name__ == "__main__":
    print(f"PUBLIC_ISSUES variables: {len(PUBLIC_ISSUES)}")
    print(f"PRIVATE_LIFE variables: {len(PRIVATE_LIFE)}")
    print(f"Total categorized: {len(PUBLIC_ISSUES) + len(PRIVATE_LIFE)}")
