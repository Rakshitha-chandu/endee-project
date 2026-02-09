from endee import Endee

client = Endee()

client.delete_index("resume_index")

client.create_index(
    name="resume_index",
    dimension=768,
    space_type="cosine"
)

print("Index reset complete")
