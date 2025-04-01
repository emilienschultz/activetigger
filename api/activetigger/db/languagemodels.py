import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from activetigger.db import DBException
from activetigger.db.models import (
    Models,
)


class LanguageModelsService:
    SessionMaker: sessionmaker[Session]

    def __init__(self, sessionmaker: sessionmaker[Session]):
        self.SessionMaker = sessionmaker

    def available_models(self, project: str):
        with self.SessionMaker() as session:
            models = session.execute(
                select(Models.name, Models.parameters, Models.path, Models.scheme_id)
                .filter_by(
                    project_id=project,
                    status="trained",
                )
                .distinct()
            ).all()
        return [
            {
                "name": m.name,
                "scheme": m.scheme_id,
                "path": m.path,
                "parameters": m.parameters,
            }
            for m in models
        ]

    def add_model(
        self,
        kind: str,
        project: str,
        name: str,
        user: str,
        status: str,
        scheme: str,
        params: dict[str, Any],
        path: str,
    ):
        session = self.SessionMaker()

        # test if the name does not exist
        models = session.query(Models).filter(Models.name == name).all()
        if len(models) > 0:
            return False

        model = Models(
            project_id=project,
            time=datetime.datetime.now(),
            kind=kind,
            name=name,
            user_id=user,
            parameters=params,
            scheme_id=scheme,
            status=status,
            path=path,
        )
        session.add(model)
        session.commit()
        session.close()

        #        print("available", self.available_models(project))

        return True

    def change_model_status(self, project: str, name: str, status: str):
        with self.SessionMaker.begin() as session:
            _ = session.execute(
                update(Models)
                .filter_by(name=name, project_id=project)
                .values(status=status)
            )

    def model_exists(self, project: str, name: str):
        session = self.SessionMaker()
        models = (
            session.query(Models)
            .filter(Models.name == name, Models.project_id == project)
            .all()
        )
        session.close()
        return len(models) > 0

    def delete_model(self, project: str, name: str):
        session = self.SessionMaker()
        # test if the name does not exist
        models = (
            session.query(Models)
            .filter(Models.name == name, Models.project_id == project)
            .all()
        )
        if len(models) == 0:
            print("Model does not exist")
            return False
        # delete the model
        session.query(Models).filter(
            Models.name == name, Models.project_id == project
        ).delete()
        session.commit()
        session.close()
        return True

    def get_model(self, project: str, name: str):
        session = self.SessionMaker()
        model = (
            session.query(Models)
            .filter(Models.name == name, Models.project_id == project)
            .first()
        )
        session.close()
        return model

    def rename_model(self, project: str, old_name: str, new_name: str):
        session = self.SessionMaker()

        # test if the name does not exist
        models = (
            session.query(Models)
            .filter(Models.name == new_name, Models.project_id == project)
            .all()
        )
        if len(models) > 0:
            raise Exception("Model already exists")
        # get and rename
        model = (
            session.query(Models)
            .filter(Models.name == old_name, Models.project_id == project)
            .first()
        )
        if model is None:
            raise DBException("Model not found")

        model.name = new_name
        model.path = model.path.replace(old_name, new_name)
        session.commit()
        session.close()
        return {"success": "model renamed"}

    def set_model_params(self, project: str, name: str, flag: str, value):
        session = self.SessionMaker()
        model = (
            session.query(Models)
            .filter(Models.name == name, Models.project_id == project)
            .first()
        )
        if model is None:
            raise DBException("Model not found")

        parameters = model.parameters.copy()
        parameters[flag] = value
        model.parameters = parameters
        session.commit()
