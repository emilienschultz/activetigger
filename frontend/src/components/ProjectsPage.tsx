import { FC } from 'react';
import { Link } from 'react-router-dom';

import { useUserProjects } from '../core/api';
import { PageLayout } from './layout/PageLayout';

import { IoIosAddCircle } from 'react-icons/io';

export const ProjectsPage: FC = () => {
  const projects = useUserProjects();

  return (
    <PageLayout currentPage="projects">
      <div className="container justify-content-center align-items-center">
        {
          <div className="row">
            <div className="col-12 col-md-8 mx-auto">
              <Link
                to="/projects/new"
                className="btn btn-primary m-3 d-flex justify-content-center align-items-center"
              >
                <IoIosAddCircle className="m-2" size={30} />
                Create new project
              </Link>

              <ul className="list-unstyled">
                {(projects || []).map((project) => (
                  <li key={project.parameters.project_name} className="projects-list">
                    <Link
                      to={`/projects/${project.parameters.project_slug}`}
                      className="project-link"
                    >
                      <b>{project.parameters.project_name}</b>
                      <br />
                      <p className="project-description">
                        (created by {project.created_by} the {project.created_at})
                      </p>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        }
      </div>
    </PageLayout>
  );
};
