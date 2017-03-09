# Play with colibricore
# Yang Xu
# 3/9/2017

import colibricore

TMPDIR = "tmp/"

corpustext = """To be, or not to be, that is the question
Whether 'tis Nobler in the mind to suffer
The Slings and Arrows of outrageous Fortune,
Or to take Arms against a Sea of troubles,
And by opposing end them? To die, to sleep
No more; and by a sleep, to say we end
The Heart-ache, and the thousand Natural shocks
That Flesh is heir to? 'Tis a consummation
Devoutly to be wished. To die, to sleep,
To sleep, perchance to Dream; Aye, there's the rub,
For in that sleep of death, what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes Calamity of so long life:
For who would bear the Whips and Scorns of time,
Th' Oppressor's wrong, the proud man's Contumely,
The pangs of despised Love, the Law’s delay,
The insolence of Office, and the Spurns
That patient merit of the unworthy takes,
When he himself might his Quietus make
With a bare Bodkin? Who would these Fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered Country, from whose bourn
No Traveler returns, Puzzles the will,
And makes us rather bear those ills we have,
Than fly to others that we know not of.
Thus Conscience does make Cowards of us all,
And thus the Native hue of Resolution
Is sicklied o'er, with the pale cast of Thought,
And enterprises of great pitch and moment,
With this regard their Currents turn awry,
And lose the name of Action. Soft you now,
The fair Ophelia. Nymph, in all thy Orisons
Be all my sins remembered"""

corpustext = corpustext.replace(',',' ,')
corpustext = corpustext.replace('.',' .')
corpustext = corpustext.replace(':',' :')

corpusfile_plaintext = TMPDIR + "hamlet.txt"
with open(corpusfile_plaintext,'w') as f:
    f.write(corpustext)


# class encoder
classfile = TMPDIR + "hamlet.colibri.cls"
#Instantiate class encoder
classencoder = colibricore.ClassEncoder()
#Save class file
classencoder.save(classfile)


# encode the corpus
corpusfile = TMPDIR + "hamlet.colibri.dat" #this will be the encoded corpus file
classencoder.encodefile(corpusfile_plaintext, corpusfile)

# decode from corpus file
classdecoder = colibricore.ClassDecoder(classfile)
decoded = classdecoder.decodefile(corpusfile)
print(decoded)


# play with patterns
p = classencoder.buildpattern("To be or not to be")
print(p.tostring(classdecoder))
print(len(p))
