import { FC } from 'react';
import { Link } from 'react-router-dom';

import { useUserProjects } from '../core/api';
import { PageLayout } from './layout/PageLayout';

import { IoIosAddCircle } from 'react-icons/io';

export const ProjectsPage: FC = () => {
  const projects = useUserProjects();

  return (
    <PageLayout currentPage="projects">
      <div className="container-fluid">
        {
          <div className="row">
            <div className="col-0 col-lg-3" />
            <div className="col-12 col-lg-6">
              <div className="w-100 d-flex align-items-center justify-content-center">
                <Link to="/projects/new" className="btn btn-warning w-75 mt-3">
                  <IoIosAddCircle className="m-1" size={30} />
                  Create new project
                </Link>
              </div>

              <ul className="list-unstyled mt-3">
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
