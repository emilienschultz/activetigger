import { FC, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

import { useUserProjects } from '../../core/api';
import { PageLayout } from '../layout/PageLayout';

import { IoIosAddCircle } from 'react-icons/io';
import { AvailableProjectsModel } from '../../types';

export const ProjectsPage: FC = () => {
  const projects = useUserProjects();
  const navigate = useNavigate();
  const [rows, setRows] = useState<AvailableProjectsModel[]>([]);
  useEffect(() => {
    setRows(projects || []);
  }, [projects]);

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const searchValue = e.target.value.toLowerCase();

    setRows(
      (projects || []).filter((project) => {
        const projectName = project.parameters.project_name.toLowerCase();
        const createdBy = project.created_by.toLowerCase();

        return projectName.includes(searchValue) || createdBy.includes(searchValue);
      }),
    );
  };

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

              <div className="project-list">
                <input
                  type="text"
                  className="form-control mt-3"
                  placeholder="Search for a project or a user"
                  onChange={handleSearch}
                />
                {rows.map((project) => (
                  <div
                    key={project.parameters.project_slug}
                    className="project-card"
                    onClick={() => navigate(`/projects/${project.parameters.project_slug}`)}
                  >
                    <h3 className="project-title">{project.parameters.project_name}</h3>
                    <p className="project-details">
                      <span>Created by: {project.created_by}</span>
                      <span>Created at: {project.created_at}</span>
                    </p>

                    <div className="badge text-bg-info" title="Memory">
                      <span className="d-none d-md-inline">HDD : {project.size} Mo</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        }
      </div>
    </PageLayout>
  );
};
