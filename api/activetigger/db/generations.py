import datetime
import logging

from sqlalchemy import select, delete
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import sessionmaker

from activetigger.datamodels import (
    GenerationAvailableModel,
    GenerationCreationModel,
    GenerationModel,
    GenerationModelApi,
)
from activetigger.db.models import GenModels, Generations


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
                .order_by(Generations.time.desc())
                .limit(n_elements)
            ).all()
            print(generated)
            return [
                [el.time, el.element_id, el.prompt, el.answer, el.endpoint]
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

    def get_project_gen_models(self, project_slug: str) -> list[GenerationModel]:
        """
        Get the GenAI model configured for the given project

        Returns a list of GenerationModel
        """
        with self.Session() as session:
            models = session.scalars(
                select(GenModels).filter_by(project_id=project_slug)
            ).all()
        return models

    def get_gen_model(self, model_id: int) -> GenerationModel:
        with self.Session() as session:
            return session.scalars(select(GenModels).filter_by(id=model_id)).first()

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
                credentials=model.credentials,
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
