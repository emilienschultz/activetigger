import datetime
import os
from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import joinedload, sessionmaker

from activetigger.datamodels import (
    GenerationAvailableModel,
    GenerationCreationModel,
    GenerationModelApi,
)
from activetigger.db.models import Generations, GenModels
from activetigger.functions import decrypt, encrypt


class GenerationsService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def add_generated(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        model_id: int,
        prompt: str,
        answer: str,
    ):
        session = self.Session()
        generation = Generations(
            user_id=user,
            time=datetime.datetime.now(),
            project_id=project_slug,
            element_id=element_id,
            model_id=model_id,
            prompt=prompt,
            answer=answer,
        )
        session.add(generation)
        session.commit()
        session.close()

    def get_generated(self, project_slug: str, username: str, n_elements: int = 10):
        """
        Get elements from generated table by order desc
        """
        with self.Session() as session:
            generated = session.scalars(
                select(Generations)
                .filter_by(project_id=project_slug, user_id=username)
                .options(joinedload(Generations.model))  # join with the model table
                .order_by(Generations.time.desc())
                .limit(n_elements)
            ).all()
            return [
                [el.time, el.element_id, el.prompt, el.answer, el.model.name]
                for el in generated
            ]

    def get_available_models(self) -> list[GenerationModelApi]:
        """
        Get the available models for generation

        Currently, this is hardwired in code
        """
        return [
            GenerationModelApi(
                name="Ollama",
                models=[
                    GenerationAvailableModel(
                        slug="llama3.1:70b", api="Ollama", name="Llama3.1 - 70b"
                    )
                ],
            ),
            GenerationModelApi(
                name="OpenAI",
                models=[
                    GenerationAvailableModel(
                        slug="gpt-4o-mini", api="OpenAI", name="ChatGPT 4o mini"
                    ),
                    GenerationAvailableModel(
                        slug="gpt-4o", api="OpenAI", name="ChatGPT 4o"
                    ),
                ],
            ),
            GenerationModelApi(name="HuggingFace", models=[]),
        ]

    def get_project_gen_models(self, project_slug: str) -> Sequence[GenModels]:
        """
        Get the GenAI model configured for the given project

        Returns a list of GenerationModel
        """
        with self.Session() as session:
            models = session.scalars(
                select(GenModels).filter_by(project_id=project_slug)
            ).all()
        return models

    def get_gen_model(self, model_id: int) -> GenModels:
        with self.Session() as session:
            result = session.scalars(select(GenModels).filter_by(id=model_id)).first()
            if result is None:
                raise Exception("Generation model not found")
            result.credentials = decrypt(result.credentials, os.environ["SECRET_KEY"])
            return result

    def add_project_gen_model(
        self, project_slug: str, model: GenerationCreationModel
    ) -> int:
        """
        Add a new GenAI model for the given project
        """
        with self.Session() as session:
            new_model = GenModels(
                project_id=project_slug,
                slug=model.slug,
                name=model.name,
                api=model.api,
                endpoint=model.endpoint,
                credentials=encrypt(model.credentials, os.environ["SECRET_KEY"]),
            )
            session.add(new_model)
            session.commit()
            session.refresh(new_model)
            return new_model.id

    def delete_project_gen_model(self, project_slug: str, model_id: int) -> None:
        """
        Delete a GenAI model from the given project
        """
        with self.Session.begin() as session:
            session.execute(
                delete(GenModels).filter_by(project_id=project_slug, id=model_id)
            )
