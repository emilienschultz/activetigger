from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from activetigger.functions import get_root_pwd, get_hash
import datetime
import json

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    project_slug = Column(String, primary_key=True)
    time_created = Column(TIMESTAMP, server_default=func.current_timestamp())
    parameters = Column(Text)
    time_modified = Column(TIMESTAMP)
    user = Column(String)

class Scheme(Base):
    __tablename__ = 'schemes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    time_created = Column(TIMESTAMP, server_default=func.current_timestamp())
    time_modified = Column(TIMESTAMP, server_default=func.current_timestamp(), onupdate=func.current_timestamp())
    user = Column(String)
    project = Column(String)
    name = Column(String)
    params = Column(Text)

class Annotation(Base):
    __tablename__ = 'annotations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    action = Column(String)
    user = Column(String)
    project = Column(String)
    element_id = Column(String)
    scheme = Column(String)
    tag = Column(String)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    user = Column(String)
    key = Column(Text)
    description = Column(Text)
    created_by = Column(String)

class Auth(Base):
    __tablename__ = 'auth'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user = Column(String)
    project = Column(String)
    status = Column(String)
    created_by = Column(String)
    

class Log(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    user = Column(String)
    project = Column(String)
    action = Column(String)
    connect = Column(String)
    

class Token(Base):
    __tablename__ = 'tokens'
    id = Column(Integer, primary_key=True, autoincrement=True)
    time_created = Column(TIMESTAMP, server_default=func.current_timestamp())
    token = Column(Text)
    status = Column(String)
    time_revoked = Column(TIMESTAMP)

class Generation(Base):
    __tablename__ = 'generations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    user = Column(String)
    project = Column(String)
    element_id = Column(String)
    endpoint = Column(String)
    prompt = Column(Text)
    answer = Column(Text)


class DatabaseManager:
    """
    Database management with SQLAlchemy
    """
    def __init__(self, path_db:str):
        self.db_url = f"sqlite:///{path_db}"

        # test if the db exists, else create it
        if not Path(path_db).exists():
            self.create_db()

        # connect the session
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.default_user = "server" 

        # check if there is a root user, add it
        session = self.Session()
        if not session.query(User).filter_by(user="root").first():
            self.create_root_session()
        session.close()

    def create_db(self):
        print("Create database")
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)

    def create_root_session(self) -> None:
        """
        Create root session
        :return: None
        """
        pwd: str = get_root_pwd()
        hash_pwd: bytes = get_hash(pwd)
        self.add_user("root", hash_pwd, "root", "system")

    def add_user(self, user:str, key:str, description:str, created_by:str):
        session = self.Session()
        user = User(user=user, key=key, description=description, created_by=created_by)
        session.add(user)
        session.commit()
        session.close()

    def add_log(self, user:str, action:str, project_slug:str, connect:str):
        session = self.Session()
        log = Log(user=user, project=project_slug, action=action, connect=connect)
        session.add(log)
        session.commit()
        session.close()

    def get_logs(self, username: str, project_slug: str, limit: int):
        session = self.Session()
        if project_slug == "all":
            logs = session.query(Log).filter_by(user=username).order_by(Log.time.desc()).limit(limit).all()
        else:
            logs = session.query(Log).filter_by(user=username, project=project_slug).order_by(Log.time.desc()).limit(limit).all()
        session.close()
        return [
        {
            'id': log.id,
            'time': log.time,
            'user': log.user,
            'project': log.project,
            'action': log.action,
            'connect': log.connect
        }
        for log in logs
    ]

    def get_project(self, project_slug: str):
        session = self.Session()
        project = session.query(Project).filter_by(project_slug=project_slug).first()
        session.close()
        if project:
            return project.__dict__
        else:
            return None
        
    def add_project(self, project_slug: str, parameters: dict, username: str):
        session = self.Session()
        project = Project(project_slug=project_slug, parameters=json.dumps(parameters), time_modified=datetime.datetime.now(), user=username)
        session.add(project)
        session.commit()
        session.close()

    def update_project(self, project_slug: str, parameters: dict):
        session = self.Session()
        project = session.query(Project).filter_by(project_slug=project_slug).first()
        project.time_modified = datetime.datetime.now()
        project.parameters = json.dumps(parameters)
        session.commit()
        session.close()
        
    def existing_projects(self) -> list:
        session = self.Session()
        projects = session.query(Project).all()
        session.close()
        return [project.project_slug for project in projects]
    
    def add_token(self, token:str, status:str):
        session = self.Session()
        token = Token(token=token, status=status)
        session.add(token)
        session.commit()
        session.close()

    def get_token_status(self, token:str):
        session = self.Session()
        token = session.query(Token).filter_by(token=token).first()
        session.close()
        if token:
            return token.status
        else:
            return None

    def revoke_token(self, token:str):
        session = self.Session()
        token = session.query(Token).filter_by(token=token).first()
        token.time_revoked = datetime.datetime.now()
        token.status = "revoked"
        session.commit()
        session.close()

    def add_scheme(self, project_slug: str, name: str, params: dict, username: str):
        session = self.Session()
        scheme = Scheme(project=project_slug, name=name, params=params, user=username)
        session.add(scheme)
        session.commit()
        session.close()

    def add_annotation(self, action: str, user: str, project_slug: str, element_id: str, scheme:str, tag:str):
        session = self.Session()
        annotation = Annotation(action=action, user=user, project=project_slug, element_id=element_id, scheme=scheme, tag=tag)
        session.add(annotation)
        session.commit()
        session.close()

    def delete_project(self, project_slug: str):
        session = self.Session()
        session.query(Project).filter(Project.project_slug == project_slug).delete()
        session.query(Scheme).filter(Scheme.project == project_slug).delete()
        session.query(Annotation).filter(Annotation.project == project_slug).delete()
        session.query(Auth).filter(Auth.project == project_slug).delete()
        session.query(Generation).filter(Generation.project == project_slug).delete()
        session.query(Log).filter(Log.project == project_slug).delete()
        session.commit()
        session.close()

