# GSS Topic Mappings for Public Issues and Private Life
# Each category is organized into narrow but coherent thematic groups

# =============================================================================
# PUBLIC_ISSUES TOPIC GROUPS
# =============================================================================

PUBLIC_ISSUES_TOPICS = {
    # =========================
    # POLITICS & IDEOLOGY
    # =========================
    'Political Ideology': [
        'polviews',  # liberal to conservative scale
    ],

    # =========================
    # REPRODUCTIVE RIGHTS
    # =========================
    'Abortion: Circumstances for Legality': [
        'abany',     # abortion for any reason
        'abdefect',  # strong chance of serious defect
        'abhlth',    # woman's health endangered
        'abnomore',  # married, wants no more children
        'abpoor',    # low income, can't afford more
        'abrape',    # pregnant from rape
        'absingle',  # not married
    ],

    'Abortion: Willingness to Help/Support': [
        'abhelp1',   # help with arrangements
        'abhelp2',   # help with paying
        'abhelp3',   # help with other costs
        'abhelp4',   # emotional support
    ],

    'Birth Control: Teen Access': [
        'pillok',    # birth control to teenagers 14-16
    ],

    # =========================
    # ENVIRONMENT
    # =========================
    'Environment: Danger Assessments': [
        'carsgen',   # car pollution danger
        'chemgen',   # pesticides danger
        'genegen',   # GMO crops danger
        'indusgen',  # industrial air pollution danger
        'nukegen',   # nuclear power danger
        'tempgen',   # greenhouse effect danger
        'tempgen1',  # climate change danger
        'watergen',  # water pollution danger
    ],

    'Environment: Personal Concern & Responsibility': [
        'grncon',    # concerned about environment
        'grneffme',  # environment affects everyday life
        'ihlpgrn',   # I do what I can for environment
        'nobuygrn',  # avoid products for environment
        'helpharm',  # hard to know if lifestyle helps
        'toodifme',  # too difficult to do anything
        'othssame',  # no point unless others do same
    ],

    'Environment: Trade-offs & Sacrifice': [
        'grnecon',   # worry too much about environment vs economy
        'grnexagg',  # environmental threats exaggerated
        'grnprice',  # pay higher prices for environment
        'grnprog',   # worry too much about progress harming environment
        'grnsol',    # accept cut in living standards
        'grntaxes',  # pay higher taxes for environment
        'impgrn',    # more important things than save environment
    ],

    'Environment: Growth & Science Solutions': [
        'grwtharm',  # economic growth always harms environment
        'grwthelp',  # America needs growth to protect environment
        'harmsgrn',  # almost everything harms environment
        'scigrn',    # modern science will solve environmental problems
    ],

    'Spending: Environment & Energy': [
        'natenvir',  # spending on improving the environment
        'natenviy',  # environment spending (version y)
        'natenrgy',  # spending on developing alternative energy sources
    ],

    # =========================
    # SCIENCE & TECHNOLOGY POLICY
    # =========================
    'Science: Research Funding': [
        'advfront',  # federal govt should support science research
        'natsci',    # spending on scientific research
    ],

    'Science: Benefits vs. Harms': [
        'balneg',    # research harmful results outweigh benefits
        'balpos',    # research benefits outweigh harmful
        'scibnfts',  # benefits outweigh harmful results
        'nanowill',  # nanotechnology benefits vs harms
    ],

    'Scientists: Public Perception': [
        'scientbe',  # scientists want to make life better
        'scientgo',  # scientists work for good of humanity
        'scienthe',  # scientists help solve problems
        'scientod',  # scientists are odd and peculiar
    ],

    'Science & Technology: Future Generations': [
        'nextgen',   # science/tech give more opportunities to next generation
    ],

    'Space Exploration Spending': [
        'natspac',   # space exploration program
        'natspacy',  # space exploration (version y)
    ],

    # =========================
    # CRIMINAL JUSTICE
    # =========================
    'Criminal Justice: Punishment': [
        'cappun',    # favor death penalty
        'courts',    # courts too harsh or not harsh enough
    ],

    'Drug Policy': [
        'grass',     # legalize marijuana
    ],

    'Police: Use of Force': [
        'polhitok',  # ever approve of police striking citizen
        'polabuse',  # citizen said vulgar things
        'polmurdr',  # citizen questioned as murder suspect
        'polescap',  # citizen attempting to escape
        'polattak',  # citizen attacking with fists
    ],

    'Spending: Crime & Drugs': [
        'natcrime',  # halting rising crime rate
        'natcrimy',  # law enforcement
        'natdrug',   # dealing with drug addiction
        'natdrugy',  # drug rehabilitation
    ],

    # =========================
    # GUNS
    # =========================
    'Gun Control': [
        'gunlaw',    # favor gun permits
    ],

    # =========================
    # RACE & EQUITY
    # =========================
    'Race: Explanations for Inequality': [
        'racdif1',   # differences due to discrimination
        'racdif2',   # differences due to inborn ability
        'racdif3',   # differences due to lack of education
        'racdif4',   # differences due to lack of will
    ],

    'Race: Government Intervention': [
        'helpblk',   # govt should aid Black people
        'natrace',   # spending on improving conditions of Black people
        'natracey',  # assistance to Black people
    ],

    'Race: Affirmative Action & Self-Help': [
        'affrmact',  # favor preference in hiring Black people
        'discaff',   # whites hurt by affirmative action
        'wrkwayup',  # Black people should work way up without favors
    ],

    'Race: Perceived Group Traits': [
        'intlwhts',  # rate whites intelligent
        'wlthwhts',  # rate whites rich/poor
        'workwhts',  # rate whites hardworking/lazy
    ],

    'Race: Housing': [
        'racopen',   # vote on open housing law
    ],

    # =========================
    # GENDER & WORKPLACE
    # =========================
    'Gender: Workplace Affirmative Action': [
        'fehire',    # hire and promote women
        'fejobaff',  # preferential hiring of women
        'discaffm',  # man won't get job/promotion
        'discaffw',  # woman won't get job/promotion
    ],

    'Gender: Traditional Roles': [
        'hubbywk1',  # men earn money, women keep house
    ],

    # =========================
    # GOVERNMENT ROLE & WELFARE
    # =========================
    'Government: Role & Size': [
        'helpnot',   # govt do more or less
    ],

    'Government: Income Redistribution': [
        'eqwlth',    # govt reduce income differences (1-7)
        'goveqinc',  # govt should reduce income differentials
        'goveqinc1', # govt should fix income differences
        'govineq1',  # politicians don't care about inequality
        'govineq2',  # govt success reducing inequality
    ],

    'Government: Welfare & Poverty': [
        'helppoor',  # govt improve standard of living
        'natfare',   # welfare spending
        'natfarey',  # assistance to the poor
    ],

    'Government: Healthcare': [
        'helpsick',  # govt help pay for medical care
        'hlthgov',   # govt provide only limited health care
        'natheal',   # improving nation's health
        'nathealy',  # health spending
    ],

    'Government: Employment & Economy': [
        'govfnanc',  # govt finance job creation projects
        'govunemp',  # govt provide unemployment benefits
    ],

    'Government: Education': [
        'govfnaid',  # govt give financial aid to students
        'nateduc',   # improving education system
        'nateducy',  # education spending
    ],

    'Government: Childcare': [
        'natchld',   # assistance for childcare
    ],

    'Spending: Cities & Infrastructure': [
        'natcity',   # solving problems of big cities
        'natcityy',  # assistance to big cities
        'natroad',   # highways and bridges
        'natmass',   # mass transportation
        'natpark',   # parks and recreation
    ],

    # =========================
    # SOCIAL SECURITY
    # =========================
    'Social Security Spending': [
        'natsoc',    # social security spending
    ],

    # =========================
    # MILITARY & FOREIGN POLICY
    # =========================
    'Military Spending': [
        'natarms',   # military spending
        'natarmsy',  # national defense
    ],

    'Foreign Aid': [
        'nataid',    # foreign aid
        'nataidy',   # assistance to other countries
        'ldctax',    # rich countries tax for poor countries
    ],

    'War Expectations': [
        'uswary',    # expect US in world war
    ],

    # =========================
    # INSTITUTIONAL CONFIDENCE
    # =========================
    'Confidence: Executive Branch': [
        'confed',    # confidence in executive branch
    ],

    'Confidence: Congress': [
        'conlegis',  # confidence in Congress
    ],

    'Confidence: Supreme Court': [
        'conjudge',  # confidence in Supreme Court
    ],

    'Confidence: Military': [
        'conarmy',   # confidence in military
    ],

    'Confidence: Business & Finance': [
        'conbus',    # confidence in major companies
        'confinan',  # confidence in banks/financial institutions
    ],

    'Confidence: Education': [
        'coneduc',   # confidence in education
    ],

    'Confidence: Organized Religion': [
        'conclerg',  # confidence in organized religion
    ],

    # =========================
    # CIVIL LIBERTIES & FREE SPEECH
    # =========================
    'Free Speech: Anti-Religious': [
        'spkath',    # allow anti-religionist to speak
        'spkathy',   # allow anti-religionist to speak (y)
        'colath',    # allow to teach
        'libath',    # allow book in library
        'libathy',   # allow book in library (y)
    ],

    'Free Speech: Homosexual': [
        'spkhomo',   # allow homosexual to speak
        'spkhomoy',  # allow homosexual to speak (y)
        'colhomo',   # allow to teach
        'libhomo',   # allow book in library
    ],

    'Free Speech: Communist': [
        'spkcom',    # allow communist to speak
        'spkcomy',   # allow communist to speak (y)
        'colcom',    # fire communist teacher
        'libcom',    # allow book in library
    ],

    'Free Speech: Racist': [
        'spkrac',    # allow racist to speak
        'spkracy',   # allow racist to speak (y)
        'colrac',    # allow to teach
        'librac',    # allow book in library
    ],

    'Free Speech: Muslim Extremist': [
        'spkmslm',   # allow Muslim preaching hatred to speak
        'spkmslmy',  # allow Muslim preaching hatred (y)
        'colmslm',   # allow to teach
        'libmslm',   # allow book in library
    ],

    'Free Speech: Militarist': [
        'spkmil',    # allow militarist to speak
        'spkmily',   # allow militarist to speak (y)
        'colmil',    # allow to teach
        'libmil',    # allow book in library
    ],

    # =========================
    # RELIGION & GOVERNMENT
    # =========================
    'Religion in Public Life': [
        'govchrst',  # govt should advocate Christian values
        'religinf',  # US better if religion had less influence
        'prayer',    # Bible prayer in public schools
    ],

    # =========================
    # EDUCATION POLICY
    # =========================
    'Sex Education': [
        'sexeduc',   # sex education in public schools
    ],

    # =========================
    # MENTAL HEALTH POLICY
    # =========================
    'Mental Health: Forced Treatment': [
        'dangroth',  # hospitalize if dangerous to others
        'dangrslf',  # hospitalize if dangerous to self
        'mustdoc',   # force examination by law
        'musthosp',  # force hospitalization by law
        'mustmed',   # force medication by law
    ],

    'Mental Health: Community Housing': [
        'viggrp',    # willing to have group home in neighborhood
    ],

    # =========================
    # FAMILY LAW
    # =========================
    'Divorce Law': [
        'divlaw',    # divorce laws easier or harder
    ],

    # =========================
    # BIOETHICS
    # =========================
    'End of Life Policy': [
        'letdie1',   # allow incurable patients to die
        'letdie1y',  # allow incurable patients to die (y)
    ],

    # =========================
    # TAXES
    # =========================
    'Federal Income Tax': [
        'tax',       # federal income tax too high/right/low
    ],

    # =========================
    # INTEREST IN POLICY ISSUES
    # =========================
    'Interest in Policy Topics': [
        'inteduc',   # interested in school issues
        'intenvir',  # interested in environmental issues
        'intfarm',   # interested in farm issues
        'intmil',    # interested in military policy
        'intspace',  # interested in space exploration
    ],

    # =========================
    # SUCCESS & MOBILITY
    # =========================
    'Perceived Barriers: Race & Gender': [
        'oprace',    # need right race to get ahead
        'opsex',     # need right sex to get ahead
    ],
}


# =============================================================================
# PRIVATE_LIFE TOPIC GROUPS
# =============================================================================

PRIVATE_LIFE_TOPICS = {
    # =========================
    # RELIGION & SPIRITUALITY
    # =========================
    'Religious Practice: Attendance & Prayer': [
        'attend',    # how often attends religious services
        'pray',      # how often prays
    ],

    'Religious Belief: God & Afterlife': [
        'god',       # confidence in existence of God
        'postlife',  # belief in life after death
        'theism',    # God concerned with humans personally
        'reborn',    # born again experience
        'relexp',    # religious experience changed life
    ],

    'Religious Belief: Bible & Meaning': [
        'bible',     # feelings about the Bible
        'godmeans',  # life meaningful because God exists
    ],

    'Religious Identity & Commitment': [
        'relidimp',  # how important is religion
        'religimp',  # how important is religion
        'relpersn',  # consider self religious person
        'sprtprsn',  # consider self spiritual person
        'relidesc',  # how well religion describes self
        'relidins',  # feel insulted if religion criticized
        'relidwe',   # how often says "we" about religion
    ],

    'Religious Social Function': [
        'makefrnd',  # religion helps make friends
        'savesoul',  # tried to convince others to accept Jesus
    ],

    'Catholic Beliefs': [
        'popespks',  # pope is infallible
    ],

    # =========================
    # SUPERNATURAL & PARANORMAL
    # =========================
    'Astrology & Paranormal': [
        'astrolgy',  # read horoscope/astrology
        'astrosci',  # is astrology scientific
    ],

    # =========================
    # SEXUAL BEHAVIOR
    # =========================
    'Sexual Partners: Count': [
        'PARTNERS_8822',  # partners in last year
        'partners',       # partners in last year
        'partnrs5',       # partners in last 5 years
        'nummen',         # male partners since 18
        'numwomen',       # female partners since 18
    ],

    'Sexual Partners: Type': [
        'acqntsex',  # sex with acquaintance
        'frndsex',   # sex with friend
        'pikupsex',  # sex with casual date
        'matesex',   # partner was spouse/regular
        'paidsex',   # sex for pay
        'othersex',  # sex with other type
    ],

    'Sexual Behavior: History': [
        'evpaidsx',  # ever paid/been paid for sex
        'evstray',   # sex other than spouse while married
    ],

    'Sexual Frequency': [
        'sexfreq',   # frequency of sex
    ],

    'Contraception': [
        'condom',    # used condom last time
    ],

    # =========================
    # SEXUAL MORALITY
    # =========================
    'Sexual Morality: Premarital': [
        'premarsx',  # sex before marriage
        'teensex',   # sex before marriage for teens 14-16
    ],

    'Sexual Morality: Extramarital': [
        'xmarsex',   # sex with person other than spouse
        'xmarsex1',  # is extramarital sex wrong
    ],

    'Sexual Morality: Homosexuality': [
        'homosex',   # homosexual sex relations
        'homosex1',  # is homosexual sex wrong
    ],

    # =========================
    # DRUG USE
    # =========================
    'Drug Use History': [
        'evcrack',   # ever used crack cocaine
        'evidu',     # ever injected drugs
    ],

    # =========================
    # WORK-LIFE BALANCE
    # =========================
    'Work Schedule Flexibility': [
        'chngtme',   # how often allowed to change schedule
        'wrkhome',   # how often works at home
        'mustwork',  # mandatory extra hours
    ],

    'Work-Family Conflict': [
        'famvswk',   # family life interferes with job
        'wkvsfam',   # job interferes with family life
        'famwkoff',  # how hard to take time off
    ],

    # =========================
    # FAMILY & PARENTING
    # =========================
    'Family Care Time': [
        'rfamlook',  # hours looking after family members
        'spfalook',  # spouse hours looking after family
    ],

    'Parenting: Discipline': [
        'spanking',  # favor spanking to discipline
    ],

    'Family Structure Views': [
        'marmakid',  # single mother can raise kids well
        'marpakid',  # single father can raise kids well
    ],

    'Gender & Family': [
        'meovrwrk',  # men hurt family when focus on work
    ],

    'Intergenerational Support': [
        'eldfnce',   # grandparents should help grandchildren financially
    ],

    # =========================
    # MARRIAGE & RELATIONSHIPS
    # =========================
    'Intermarriage Attitudes': [
        'marasian',  # close relative marry Asian
        'marwht',    # close relative marry white
        'relmarry',  # accept different religion marrying relative
    ],

    'Marital Happiness': [
        'hapmar',    # happiness of marriage
        'hapcohab',  # happiness with partner
    ],

    # =========================
    # HAPPINESS & LIFE SATISFACTION
    # =========================
    'General Happiness': [
        'happy',     # general happiness
        'happy7',    # how happy
        'hapunhap',  # happy or unhappy with life
    ],

    'Life Assessment': [
        'life',      # is life exciting or dull
    ],

    'Domain Satisfaction': [
        'satfam7',   # family satisfaction
        'satfin',    # financial satisfaction
        'satjob',    # job satisfaction
    ],

    # =========================
    # LIFE PHILOSOPHY & MEANING
    # =========================
    'Life Meaning & Purpose': [
        'egomeans',  # life meaningful only if you provide meaning
        'nihilism',  # life serves no purpose
    ],

    'Agency & Fatalism': [
        'fatalism',  # people can't change course of their lives
    ],

    # =========================
    # TRUST & SOCIAL ATTITUDES
    # =========================
    'Social Trust': [
        'helpful',   # people helpful or looking out for selves
        'helpfulnv', # people helpful (no vol)
        'helpfulv',  # people helpful (with vol)
    ],

    'Fear & Safety': [
        'fear',      # afraid to walk at night in neighborhood
    ],

    'Mental Health Stigma': [
        'fammhneg',  # family's negative attitudes about mental health
        'othmhneg',  # others' negative attitudes about mental health
    ],

    # =========================
    # VALUES & PRIORITIES
    # =========================
    'Personal Values: Self vs. Others': [
        'firstyou',  # take care of self first
        'helpfrds',  # better off should help less well-off friends
    ],

    # =========================
    # BELIEFS ABOUT SUCCESS
    # =========================
    'Success Factors: Education & Effort': [
        'opeduc',    # need good education to get ahead
        'ophrdwrk',  # need to work hard to get ahead
    ],

    'Success Factors: Social Capital': [
        'opclout',   # need political connections
        'opknow',    # need to know right people
    ],

    'Success Factors: Background': [
        'oppared',   # need educated parents
        'opwlth',    # need wealthy family
        'oprelig',   # need right religion
    ],

    # =========================
    # LIFE OUTLOOK
    # =========================
    'Economic Optimism': [
        'goodlife',  # standard of living will improve
        'parsol',    # living standard compared to parents
        'kidssol',   # kids' living standard compared to self
    ],

    # =========================
    # LEISURE & SOCIAL ACTIVITIES
    # =========================
    'Social Time: Family & Friends': [
        'socrel',    # spend evening with relatives
        'socfrend',  # spend evening with friends
        'socommun',  # spend evening with neighbor
    ],

    'Leisure: Entertainment': [
        'socbar',    # spend evening at bar
        'xmovie',    # seen x-rated movie
    ],

    'Leisure: Outdoor & Cultural': [
        'hunt',      # do you hunt
        'visnhist',  # visited natural history museum
        'vissci',    # visited science museum
        'viszoo',    # visited zoo
    ],

    'Relaxation Time': [
        'hrsrelax',  # hours per day to relax
    ],

    # =========================
    # DIGITAL LIFE
    # =========================
    'Internet & Email Usage': [
        'emailhr',   # email hours per week
        'emailmin',  # email minutes per week
        'wwwhr',     # www hours per week
        'wwwmin',    # www minutes per week
    ],

    # =========================
    # NEIGHBORHOOD
    # =========================
    'Neighborhood Composition': [
        'raclive',   # any opposite race in neighborhood
    ],

    # =========================
    # WORK ATTITUDES
    # =========================
    'Work Motivation': [
        'richwork',  # if rich, continue or stop working
    ],

    # =========================
    # SCIENCE & RELIGION
    # =========================
    'Science & Change': [
        'toofast',   # science makes life change too fast
        'trustsci',  # we trust too much in science
    ],

    # =========================
    # PERSONAL INTERESTS
    # =========================
    'Interest in Science & Technology': [
        'intmed',    # interested in medical discoveries
        'intsci',    # interested in scientific discoveries
        'inttech',   # interested in technologies
    ],

    # =========================
    # ENVIRONMENTAL HABITS
    # =========================
    'Recycling Behavior': [
        'recycle',   # how often recycles
    ],
}


# Print summary
if __name__ == "__main__":
    public_vars = sum(len(v) for v in PUBLIC_ISSUES_TOPICS.values())
    private_vars = sum(len(v) for v in PRIVATE_LIFE_TOPICS.values())

    print("PUBLIC_ISSUES_TOPICS:")
    print(f"  Groups: {len(PUBLIC_ISSUES_TOPICS)}")
    print(f"  Variables: {public_vars}")
    print()
    print("PRIVATE_LIFE_TOPICS:")
    print(f"  Groups: {len(PRIVATE_LIFE_TOPICS)}")
    print(f"  Variables: {private_vars}")
    print()
    print(f"Total: {len(PUBLIC_ISSUES_TOPICS) + len(PRIVATE_LIFE_TOPICS)} groups, {public_vars + private_vars} variables")
