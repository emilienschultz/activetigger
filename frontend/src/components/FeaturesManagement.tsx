import cx from 'classnames';
import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { Modal } from 'react-bootstrap';
import { FaPlusCircle } from 'react-icons/fa';
import { useDeleteFeature, useGetFeatureInfo } from '../core/api';
import { useAppContext } from '../core/context';
import { sortDatesAsStrings } from '../core/utils';
import { FeatureDescriptionModelOut } from '../types';
import { CreateNewFeature } from './forms/CreateNewFeature';
import { ModelsPillDisplay } from './ModelsPillDisplay';

export default function SimpleTable(data: FeatureDescriptionModelOut) {
  return (
    <table id="parameter-tables">
      <tbody>
        <tr>
          <td className="key">Name</td>
          <td className="value">{data.name}</td>
        </tr>
        <tr>
          <td className="key">User</td>
          <td className="value">{data.user}</td>
        </tr>
        <tr>
          <td className="key">Time</td>
          <td className="value">{data.time}</td>
        </tr>
        <tr>
          <td className="key">Kind</td>
          <td className="value">{data.kind}</td>
        </tr>
        <tr>
          <td className="key last-row">Parameters</td>
          <td className="value last-row">{JSON.stringify(data.parameters, null, 2)}</td>
        </tr>
      </tbody>
    </table>
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

  if (!project) {
    return <div>No project selected</div>;
  }
  return (
    <div className="row">
      <ModelsPillDisplay
        modelNames={Object.values(featuresInfo || {})
          .sort((featureA, featureB) => sortDatesAsStrings(featureA?.time, featureB?.time, true))
          .map((feature) => (feature && feature.name ? feature.name : ''))}
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
            Currently computing {element ? element.name : ''}
            {element?.progress ? ` (${element.progress}%)` : ''}
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
  );
};
