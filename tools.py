from config import config
from remote_db import Index

class NLAChain:
    def __init__(self):
        from langchain.chat_models import ChatOpenAI
        self.llm = ChatOpenAI(temperature=0)

        self.toolkits = self.get_toolkits()

    def store_plugins(self, ai_plugins):
        from langchain.schema import Document
        docs = [
            Document(
                page_content=plugin.description_for_model,
                metadata={
                    "plugin_name": plugin.name_for_model
                }
            )
            for plugin in ai_plugins
        ]
        Index.local_storage(docs, "ai-plugins")

    def get_toolkits(self):
        urls = [
            "https://datasette.io/.well-known/ai-plugin.json",
            "https://api.speak.com/.well-known/ai-plugin.json",
            "https://www.wolframalpha.com/.well-known/ai-plugin.json",
            "https://www.zapier.com/.well-known/ai-plugin.json",
            "https://www.klarna.com/.well-known/ai-plugin.json",
            "https://www.joinmilo.com/.well-known/ai-plugin.json",
            "https://slack.com/.well-known/ai-plugin.json",
            "https://schooldigger.com/.well-known/ai-plugin.json",
        ]
        from langchain.tools.plugin import AIPlugin
        AI_PLUGINS = [AIPlugin.from_url(url) for url in urls]

        from langchain.agents.agent_toolkits import NLAToolkit
        toolkits_dict = {
            plugin.name_for_model:NLAToolkit.from_llm_and_ai_plugin(
                self.llm, plugin
            )
            for plugin in AI_PLUGINS
        }
        # self.store_plugins(AI_PLUGINS)
        return toolkits_dict

    def get_tools(self, input_text):
        # Get documents, which contain the Plugins to use
        res = Index.query_local_index(input_text, "ai-plugins")

        # Get the toolkits, one for each plugin
        tool_kits = [self.toolkits[d.metadata["plugin_name"]] for d in res]

        # Get the tools: a separate NLAChain for each endpoint
        tools = []
        for tk in tool_kits:
            tools.extend(tk.nla_tools)
        return tools

if __name__ == "__main__":
    config()
    nla = NLAChain()
    nla.get_toolks("Translate English to Chinese.")