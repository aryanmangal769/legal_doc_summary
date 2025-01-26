from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the models
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True )
base_model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True )
fine_tuned_model = AutoModelForCausalLM.from_pretrained("./fine_tuned_lora_model", load_in_8bit=True )
fine_tuned_model_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_lora_model", load_in_8bit=True )

# Initialize summarization pipeline for both models
base_model_pipeline = pipeline("text-generation", model=base_model, tokenizer=base_model_tokenizer)
fine_tuned_model_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_model_tokenizer)

# Legal document to summarize
legal_document = """

Summarize the following legal text:
one lakshminarayana iyer a hindu brahmin who owned considerable properties in the tirunelveli district died on 13th december 1924 leaving him surviving a widow ranganayaki and a married daughter ramalakshmi.
ramalakshmi had married the plaintiff and had a number of children from him.
they were all alive in december 1924 when lakshminarayana died.
before his death he executed a will on 16th november 1924 the construction of which is in controversy in this appeal.
by this will he gave the following directions after my lifetime you the aforesaid ranganayaki amminal my wife shall till your lifetime enjoy the aforesaid entire properties the outstandings due to me the debts payable by me and the chit amounts payable by me.
after your lifetime ramalakshmi ammal our daughter and wife of rama ayyar avergal of melagaram village and her heirs shall enjoy them with absolute rights and powers of alienation such as gift exchange and sale from son to grandson and so on for generations.
as regards the payment of maintenance to be made to chinnanmal alias lakshmi ammal wife of my late son hariharamayyan my wife ranganayaki ammal shall pay the same as she pleases and obtain a release deed.
ranganayaki entered into possession of the properties on the death of her husband.
on 21st february 1928 she settled the maintenance claim of lakshmi ammal and obtained a deed of release from her by paying her a sum of rs 3350 in cash and by executing in her favour an agreement stipulating to pay her a sum of rs 240 per annum.
ramalakshmi died on 25th april 1938 during the lifetime of the widow.
none of her children survived her.
on the 24th july 1945 the widow describing herself as an absolute owner of the properties of her husband sold one of the items of the property to the 2nd defendant for rs 500.
on the 18th september 1945 the suit out of which this appeal arises was instituted by the plaintiff the husband and the sole heir of ramalakshmi for a declaration that the said sale would not be binding on him beyond the lifetime of the widow.
a prayer was made that the widow be restrained from alienating the other properties in her possession.
on the 19th september 1945 an ad interim injunction was issued by the high court restraining the widow from alienating the properties in her possession and forming part of her husband 's estate inspite of this injunction on the 27th september 1945 she executed two deeds of settlement in favour of the other defendants comprising a number of properties.
the plaintiff was allowed to amend his plaint and include therein a prayer for a declaration in respect of the invalidity of these alienations as well.
it was averred in the plaint that ramalakshmi obtained a vested interest in the suit properties under the will of her father and plaintiff was thus entitled to maintain the suit.
the defendants pleaded that the plaintiff had no title to maintain the suit that the widow was entitled under the will to an absolute estate or at least to an estate analogous to and not less than a widow 's estate that the estate given to ramalakshmi under the will was but a contingent one and she having predeceased the widow no interest in the suit properties devolved on the plaintiff.
the main issue in the suit was whether the widow took under the will an absolute estate or an estate like the hindu widow 's estate and whether the daughter 's interest therein was in the nature of a contingent remainder or whether she got in the properties a vested interest.
the subordinate judge held that the widow took under the will a limited life interest and not an absolute estate or even a widow 's estate under hindu law and that the daughter got there under a vested interest in the properties to which the plaintiff succeeded on her death.
in view of this finding he granted the plaintiff a declaratory decree to the effect that the first defendant had only an estate for life in the suit properties and that the alienations made by her would not endure beyond her lifetime.
the question as to the validity of the alienations was left undetermined.
the unsuccessful defendants preferred an appeal against this decree to the high court of judicature at madras.
during the pendency of the appeal the widow died on 14th february 1948.
the high court by its judgment under appeal affirmed the decision of the trial judge and maintained his view on the construction of the will.
leave to appeal to the supreme court was granted and the appeal was admitted on the 27th november 1951.
the substantial question to decide in the appeal is whether the estate granted by the testator to his widow was a fall woman 's estate under hindu law or merely a limited life estate in the english sense of that expression.
it was not contested before us that a hindu can by will create a life estate or successive life estates or any other estate for a limited term provided the donee or the persons taking under it are capable of taking under a deed or will.
the decision of the appeal thus turns upon the question whether the testator 's intention was to give to his widow ail ordinary life estate or an estate analogous to that of a hindu widow.
at one time it was a moot point whether a hindu widow 's estate could be created by will it being an estate created by law but it is now settled that a hindu can confer by means of a will oil his widow the same estate which she would get by inheritance.
the widow in such a case takes as a demise and not as an heir.
the court 's primary duty in such cases is to ascertain from the language employed by the testator what were his intentions keeping in view the surrounding circumstances his ordinary notions as a hindu in respect to devolution of his property his family relationships etc.
in other words to ascertain his wishes by putting itself so to say in his armchair.
considering the will in the light of these principles it seems to us that lakshminarayan iyer intended by his will to direct that his entire properties should be enjoyed by his widow during her lifetime but her interest in these properties should come to an end on her death that all these properties in their entirety should thereafter be enjoyed as absolute owners by his daughter and her heirs with powers of alienation gift exchange and sale from generation to generation.
he wished to make his daughter a fresh stock of descent so that her issue male or female may have the benefit of his property.
they were the real persons whom he earmarked with certainty as the ultimate recipients of his bounty.
in express terms he conferred on his daughter powers of alienation byway of gift exchange sale but in sharp contrast to this on his widow he conferred no such powers.
the direction to her was that she should enjoy the entire properties including the outstandings etc.
and these shall thereafter pass to her daughters.
though no restraint in express terms was put on her powers of alienation in case of necessity even that limited power was not given to her in express terms.
if the testator had before his mind 's eye his daughter and her heirs as the ultimate beneficiaries of his bounty that intention could only be achieved by giving to the widow a limited estate because by conferring a full hindu widow 's estate on her the daughter will only have a mere spes successions under the hindu law which may or may not mature and under the will her interest would only be a contingent one in what was left indisposed of by the widow.
it is significant that the testator did not say in the will that the daughter will enjoy only the properties left indisposed of by the widow.
the extent of the grant so far as the properties mentioned in the schedule are concerned to the daughter and the widow is the same.
just as the widow was directed to enjoy tile entire properties mentioned in the schedule during her lifetime in like manner the daughter and her heirs were also directed to enjoy the same properties with absolute rights from generation to generation.
they could not enjoy the same properties in the manner directed if the widow had a full hindu widow 's estate and had the power for any purpose to dispose of them and did so.
if that was the intention the testator would clearly have said that the daughter would only take the properties remaining after the death of the widow.
the widow can not be held to have been given a full hindu widow 's estate under the will unless it can be said that under its terms she was given the power of alienation for necessary purposes whether in express terms or by necessary implication.
as above pointed out admittedly power of alienation in express terms was not conferred on her.
it was argued that such a power was implicit within the acts she was authorized to do that is to say when she was directed to pay the debts and settle the maintenance of ramalakshmi it was implicit within these directions that for these purposes if necessity arose she could alienate the properties.
this suggestion in the surrounding circumstances attending the execution of this will can not be sustained.
the properties disposed of by the will and mentioned in the schedule were considerable in extent and it seems that they fetched sufficient income to enable the widow to fulfil the obligations under the will.
indeed we find that within four years of the death of the testator the widow was able to pay a lump sum of rs 3350 in cash to the daughter in law without alienating any part of the immovable properties and presumably by this time she had discharged all the debts.
it is not shown that she alienated a single item of immovable property till the year 1945 a period of over 21 years after the death of her husband excepting one which she alienated in the year 1937 to raise a sum of rs 1000 in order to buy some land.
by this transaction she substituted one property by another.
for the purpose of her maintenance for payment of debts etc and for settling the claim of the daughter in law she does not appear to have felt any necessity to make any alienation of any part of the estate mentioned in the schedule and the testator in all likelihood knew that she could fulfil these obligations without having recourse to alienations and hence he did not give her any power to do so.
in this situation the inference that the testator must have of necessity intended to confer on the widow power of alienation for those limited purposes can not be raised.
in our opinion even if that suggestion is accepted that for the limited purposes mentioned in the will the widow could alienate this power would fall far short of the powers that a hindu widow enjoys under hindu law.
under that law she has the power to alienate the estate for the benefit of the soul of the husband for pilgrimage and for the benefit of the estate and for other authorized purposes.
it can not be said that a hindu widow can only alienate her husband 's estate for payment of debts to meet maintenance charges and for her own maintenance.
she represents the estate in all respects and enjoys very wide power except that she can not alienate except for necessity and her necessities have to be judged on a variety of considerations.
we therefore hold that the estate conferred on ranganayaki ammal was more like the limited estate in the english sense of the term than like a full hindu widow 's estate in spite of the directions above mentioned.
she had complete control over the income of the property during her lifetime.
but she had no power to deal with the corpus of the estate.
and it had to be kept intact for the enjoyment of the daughter.
though the daughter was not entitled to immediate possession of the property it was indicated with certainty that she should get the entire estate at the proper time.
and she thus got an interest in it on the testator 's death.
she was given a present right of future enjoyment in the property.
according to jarman jarman on wills the law leans in favour of vesting of estates and the property disposed of belongs to the object of the gift when the will takes effect and we think the daughter got under this will a vested interest in the testator 's properties on his death.
it was strenuously argued by mr k section krishnaswami iyengar that lakshminarayana iyer was a brahmin gentleman presumably versed in the sastras living in a village in the southernmost part of the madras state that his idea of a restricted estate was more likely to be one analogous to a hindu woman 's estate than a life estate a understood in english law wherein the estate is measured by use and not by duration and that if this will was construed in the light of the notions of lakshminarayana iyer it should be held that the widow got under it a hindu widow 's estate and the daughter got under it a contingent remainder in the nature of spes and on her death there was nothing which could devolve on the plaintiff and he thus had no locus standi to question the alienations made by the widow.
the learned counsel in support of his contention drew our attention to a number of decisions of different high courts and contended that the words of this will should be construed in the manner as more or less similar words were construed by the courts in the wills dealt with in those decisions.
this rule of construction by analogy is a dangerous one to follow in construing wills differently worded and executed in different surroundings.
vide sasiman v shib narain 491.
a 2 5.
however out of respect for learned counsel on both sides who adopted the same method of approach we proceed to examine some of the important cases referred to by them.
mr krishnaswami iyengar sought to derive the greatest support for his contention from the decision in ram bahadur v jager.
nath prasad 3 pat.
l j 199.
the will there recited that if a daughter or son was born to the testator during his lifetime such son or daughter would be the owner of all his properties but if there was no son or daughter his niece section would get a bequest of a lakh of rupees and the rest of the movable and immovable properties would remain in possession of his wife until her death and after her these would remain in possession of his niece.
the remainder was disposed of in the following words if on the death of my wife and my niece there be living a son and a daughter born of the womb of my said brother 's daughter then two thirds of the movable property will belong to the son and one third to the daughter.
but as regards the immovable property none shall have the lest right of alienation.
they will of course be entitled to enjoy the balance left after payment of rent.
this will was construed as conveying an absolute estate to the son and the daughter of the niece.
it was remarked that in spite of an express restriction against alienation the estate taken by section the niece was an estate such as a woman ordinarily acquires by inheritance under the hindu law which she holds in a completely representative character but is unable to alienate except in case of legal necessity and that such a construction was in accordance with the ordinary notions that a hindu has in regard to devolution of his property.
the provisions contained in this will bear no analogy to those we have to construe.
the restraint against alienation was repugnant to both a life estate and a widow estate and was not therefore taken into account.
but there were other indications in that will showing that a widow 's estate had been given.
the fact that the gift over was a contingent bequest was by itself taken as a sure indication that the preceding bequest was that of a widow 's estate.
there is no such indication in the will before us.
reliance was next placed on the decision in pavani subbamma v ammala rama naidu 1937.
1 m l j 268.
1936 indlaw mad 236.
'
"""

# Function to summarize using a model
def summarize_with_model(model_pipeline, document, max_length=512):
    # Generate summary from model
    generated_summary = model_pipeline(document, max_length=10000, num_return_sequences=1)[0]['generated_text']
    return generated_summary

# Generate summaries using both models
base_model_summary = summarize_with_model(base_model_pipeline, legal_document)
fine_tuned_model_summary = summarize_with_model(fine_tuned_model_pipeline, legal_document)

# Output the summaries
print("Base Model Summary:")
print(base_model_summary)

print("\nFine-Tuned Model Summary:")
print(fine_tuned_model_summary)
