import { FC, useState } from 'react';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';

import { Modal } from 'react-bootstrap';
import { useDeleteFeature, useGetFeatureInfo } from '../core/api';
import { useAppContext } from '../core/context';
import { CreateNewFeature } from './CreateNewFeature';

export const FeaturesManagement: FC = () => {
  const { projectName } = useParams();

  // get element from the state
  const {
    appContext: { currentProject: project, isComputing },
  } = useAppContext();

  // API calls
  const { featuresInfo } = useGetFeatureInfo(projectName || null, project);
  const deleteFeature = useDeleteFeature(projectName || null);

  // show the menu
  const [showMenu, setShowMenu] = useState(false);

  const deleteSelectedFeature = async (element: string) => {
    await deleteFeature(element);
  };

  if (!project) {
    return <div>No project selected</div>;
  }

  return (
    <div className="container">
      <div className="row">
        {featuresInfo &&
          Object.entries(featuresInfo).map(([key, value]) => (
            <div className="card text-bg-light mt-3" key={key}>
              <div className="d-flex m-2 align-items-center">
                <button
                  className="btn btn p-0 mx-4"
                  onClick={() => {
                    deleteSelectedFeature(key);
                  }}
                >
                  <MdOutlineDeleteOutline size={20} />
                </button>
                <span className="w-25">{key}</span>
                <span className="mx-2">{value?.time}</span>
                <span className="mx-2">by {value?.user}</span>
                {value?.kind === 'regex' && <span>N={value.parameters['count'] as string}</span>}
              </div>
            </div>
          ))}{' '}
        {/* Display computing features */}
        {Object.entries(project?.features.training).map(([key, element]) => (
          <div className="card text-bg-light mt-3 bg-warning" key={key}>
            <div className="d-flex m-2 align-items-center">
              <span className="w-25">
                Currently computing {element ? element.name : ''}
                {element?.progress ? ` (${element.progress}%)` : ''}
              </span>
            </div>
          </div>
        ))}
        {/* // create new feature */}
        <button
          onClick={() => setShowMenu(!showMenu)}
          className="btn btn-primary w-25 mt-3"
          disabled={isComputing}
        >
          Create a new feature
        </button>
        <Modal show={showMenu} onHide={() => setShowMenu(false)} id="addfeature-modal">
          <Modal.Header closeButton>
            <Modal.Title>Add a new feature</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <CreateNewFeature
              columns={project?.params.all_columns || []}
              featuresOption={project.features.options || {}}
            />
          </Modal.Body>
        </Modal>
      </div>
    </div>
  );
};
