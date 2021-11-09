import argparse
import time
from discord import Status
from discord import Game
from discord.ext.commands import Bot
from discord import Embed
from discord import Colour

from storytpu import *

start_time = time.time()

parser = argparse.ArgumentParser(description='Input argument parser.')

parser.add_argument('--token', type=str, help='Bot token. Do not share this!')
parser.add_argument('--prefix', type=str, help='Prefix for interacting with the bot.', default='y!')

parser.add_argument('--openai', type=str, help='OpenAI API Key. Do NOT share this!')
parser.add_argument('--openai_engine', type=str, help='OpenAI GPT-3 Engine Backend', default='davinci')
parser.add_argument('--model_name', type=str, help='HuggingFace model name', default='hakurei/c1-6B')
parser.add_argument('--top_p', type=float, help='cut off probablity for nucleus sampling', default=1.0)
parser.add_argument('--top_k', type=float, default=120)
parser.add_argument('--rep_p', type=float, help='Repetition penalty to prevent model from repeating tokens', default=1.2)
parser.add_argument('--rep_p_slope', type=float, help='Repetition penalty slope.', default=0.18)
parser.add_argument('--tfs', type=float, help='Tail free sampling', default=0.9)
parser.add_argument('--temperature', type=float, default=0.53)
parser.add_argument('--output_length', type=int, help='length of output sequence (number of tokens)', default=50)
parser.add_argument('--past_length', type=int, help='amount of memorized inputs and responses', default=5)
parser.add_argument('--mem_path', type=str, help='path to memories json file', default='memory.json')
args = parser.parse_args()

client = Bot(command_prefix=args.prefix)

embed_colour = Colour.from_rgb(178, 0, 255)

def trunc(num, digits):
        sp = str(num).split('.')
        return '.'.join([sp[0], sp[1][:digits]])

def log(com, logstr):
        timestamp = time.time() - start_time
        print('[' + trunc(timestamp, 4) + '] ' + com + ': ' + logstr)

def actjob(message, author):
        while args.is_generating:
                time.sleep(0)
        args.is_generating = True
        if message != '':
                log('ai  ', 'Processing Act job -- [' + message + ']')
        else:
                log('ai  ', 'Processing Redo job')

        args.input_stack = []
        args.context = ''
        start_time = time.time()
        output = run_model(args, message)
        elapsed_time = time.time() - start_time
        logged_output = output.replace('\n', '')

        log('ai  ', 'Generated result -- [' + logged_output + ']')

        embed = Embed(description='```' + output + '```', colour=embed_colour)
        embed.set_footer(text='{:.3f} seconds'.format(elapsed_time), icon_url=author.avatar_url)
        
        args.is_generating = False

        return embed

@client.event
async def on_ready():
        log('init', 'Connected to Discord servers.')
        game = Game('with Google\'s electricity bills.')
        await client.change_presence(status=Status.online, activity=game)

@client.command(name='temp',
                description='Sets the temperature for the AI. Higher number == More random results.',
                brief='Sets the temperature for the AI.',
                pass_context=True)
async def tempcmd(context):
        log('ai  ', 'Changing temperature to ' + context.message.content[7:])
        args.temperature = float(context.message.content[7:])

@client.command(name='rep_p',
                description='Penalize repeated tokens. Higher number == Less repetition',
                brief='Penalize repetition.',
                pass_context=True)
async def rep_pcmd(context):
        log('ai  ', 'Changing Rep_P to ' + context.message.content[8:])
        args.rep_p = float(context.message.content[8:])

@client.command(name='rep_p_slope',
                brief='Set the Repetition Penalty Slope',
                pass_context=True)
async def rep_slopecmd(context):
        log('ai  ', 'Changing Rep_P_Slope to ' + context.message.content[15:])
        args.rep_p_slope = float(context.message.content[15:])

@client.command(name='tfs',
                brief='Tail Free Sampling',
                pass_context=True)
async def tfscmd(context):
        log('ai  ', 'Changing TFS to ' + context.message.content[6:])
        args.tfs = float(context.message.content[6:])

@client.command(name='top_p',
                description='Adjust the summed probabilities of what tokens should considered to be generated.',
                brief='Use this with Nucleus sampling to get rid of neural degeneration.',
                pass_context=True)
async def top_pcmd(context):
        log('ai  ', 'Changing Top_P to ' + context.message.content[8:])
        args.top_p = float(context.message.content[8:])

@client.command(name='outlength',
                description='Change length of generated output',
                brief='Output length',
                pass_context=True)
async def outlengthcmd(context):
        log('ai  ', 'Changing output_length to ' + context.message.content[12:])
        args.output_length = int(context.message.content[12:])

@client.command(name='phrase',
                brief='Add phrase to biased phrases.',
                pass_context=True)
async def phrasecmd(context):
        phrase = context.message.content[9:]
        args.phrase_biases.append(phrase)

@client.command(name='ban',
                brief='Add phrase to banned phrases.',
                pass_context=True)
async def bancmd(context):
        phrase = context.message.content[6:]
        args.banned_phrases.append(phrase)

@client.command(name='delbias',
                brief='Delete phrase biases',
                pass_context=True)
async def delbias(context):
        args.phrase_biases = []

@client.command(name='delban',
                brief='Delete banned phrases',
                pass_context=True)
async def delban(context):
        args.banned_phrases = []

@client.command(name='bias',
                brief='Set the bias for the phrases',
                pass_context=True)
async def biascmd(context):
        log('ai  ', 'Changing bias to ' + context.message.content[7:])
        args.bias = float(context.message.content[7:])

@client.command(name='gen',
                description='Generate example from prompt',
                brief='Generate',
                pass_context=True)
async def gencmd(context):
        genmsg = context.message.content[6:]
        await context.message.channel.send(embed=actjob(genmsg, context.message.author))

@client.command(name='params',
                description='List AI parameters',
                brief='List params',
                pass_context=True)
async def paramscmd(context):
        description = "__Processing Parameters__\n\nGPT-J Engine Backend = ``" + args.model_name + "``\nOutput Length = ``" + str(args.output_length) + "``\nTop_P = ``" + str(args.top_p) + "``\nTFS = ``" + str(args.tfs) + "``\nRep_P = ``" + str(args.rep_p) + "``\nRep_P_Slope = ``" + str(args.rep_p_slope) + "``\nTemperature = ``" + str(args.temperature) + "``\n"
        await context.message.channel.send(embed=Embed(title='AI Parameters', description=description, colour=embed_colour))

def main():
        if not args.token:
                print('token must be provided')
                exit()

        log('init', 'Server started at ' + time.strftime("%Y-%m-%d %H:%M"))
        log('init', 'Loading the model.')
        args.is_generating = False
        init_model(args)
        client.run(args.token)

if __name__ == '__main__':
        main()
