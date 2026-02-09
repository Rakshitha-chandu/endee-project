from endee import Endee, Precision

client = Endee()

client.create_index(
    name="resume_index",
    dimension=768,
    space_type="cosine",
    precision=Precision.INT8D
)
