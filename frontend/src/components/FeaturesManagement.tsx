import cx from 'classnames';
import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { Modal } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import { useDeleteFeature, useGetFeatureInfo } from '../core/api';
import { useAppContext } from '../core/context';
import { FeatureDescriptionModelOut } from '../types';
import { CreateNewFeature } from './CreateNewFeature';
import { ModelsPillDisplay } from './ModelsPillDisplay';

export default function SimpleTable(data: FeatureDescriptionModelOut) {
  return (
    <div>
      <table className="table-auto border-collapse border border-gray-300 w-full">
        <tbody>
          <tr>
            <td className="border border-gray-300 px-4 py-2 font-medium">Name</td>
            <td className="border border-gray-300 px-4 py-2">{data.name}</td>
          </tr>
          <tr>
            <td className="border border-gray-300 px-4 py-2 font-medium">User</td>
            <td className="border border-gray-300 px-4 py-2">{data.user}</td>
          </tr>
          <tr>
            <td className="border border-gray-300 px-4 py-2 font-medium">Time</td>
            <td className="border border-gray-300 px-4 py-2">{data.time}</td>
          </tr>
          <tr>
            <td className="border border-gray-300 px-4 py-2 font-medium">Kind</td>
            <td className="border border-gray-300 px-4 py-2">{data.kind}</td>
          </tr>
          <tr>
            <td className="border border-gray-300 px-4 py-2 font-medium">Parameters</td>
            <td className="border border-gray-300 px-4 py-2">
              {JSON.stringify(data.parameters, null, 2)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

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
  const [showAddFeature, setShowAddFeature] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const deleteSelectedFeature = async (element: string) => {
    await deleteFeature(element);
    setSelectedFeature(null);
  };

  console.log(featuresInfo);

  if (!project) {
    return <div>No project selected</div>;
  }
  return (
    <div className="container">
      <div className="row">
        <ModelsPillDisplay
          modelNames={Object.keys(featuresInfo || {}).map((feature) => feature)}
          currentModelName={selectedFeature}
          setCurrentModelName={setSelectedFeature}
          deleteModelFunction={deleteSelectedFeature}
        >
          <button
            onClick={() => setShowAddFeature(true)}
            className={cx('model-pill ', isComputing ? 'disabled' : '')}
            id="create-new"
          >
            <FaPlusCircle size={20} /> Create new feature
          </button>
        </ModelsPillDisplay>
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
        {featuresInfo &&
          selectedFeature &&
          SimpleTable(featuresInfo[selectedFeature] as FeatureDescriptionModelOut)}

        <Modal show={showAddFeature} onHide={() => setShowAddFeature(false)} id="addfeature-modal">
          <Modal.Header closeButton>
            <Modal.Title>Add a new feature</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <CreateNewFeature
              columns={project?.params.all_columns || []}
              featuresOption={project.features.options || {}}
              callback={setShowAddFeature}
            />
          </Modal.Body>
        </Modal>
      </div>
    </div>
  );
};
